"""Message passing, network-jitter emulation and the CPU-bound task used by the fault-tolerant balancer.

Framing: every message is a JSON object sent as one frame (4-byte big-endian length + UTF-8 body) over
TCP on the loopback interface. TCP gives reliable FIFO channels, the channel model of Lynch (1996, ch. 14).

Jitter: every outgoing frame passes through a JitterSender, which holds it for a random delay before it
reaches the socket. Departure times are made non-decreasing (d_k = max(d_{k-1}, now + delay)), so a long
delay also holds back the frames behind it, as head-of-line blocking does on a real TCP connection.
"""
import json
import math
import os
import random
import socket
import struct
import threading
import time
from collections import deque

HDR = struct.Struct("!I")
HOST = "127.0.0.1"

# ---------------------------------------------------------------------------------------------- framing


def encode(obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    return HDR.pack(len(body)) + body


def _exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv(sock):
    """Next message and its size on the wire, or (None, 0) once the peer has closed the connection."""
    hdr = _exact(sock, HDR.size)
    if hdr is None:
        return None, 0
    n = HDR.unpack(hdr)[0]
    body = _exact(sock, n)
    return (None, 0) if body is None else (json.loads(body), n + HDR.size)


def listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, 0))
    s.listen(64)
    return s


def connect(port, retries=500):
    for _ in range(retries):
        try:
            s = socket.create_connection((HOST, port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except ConnectionRefusedError:
            time.sleep(0.01)
    raise ConnectionError(f"could not connect to port {port}")


def now_ns():
    """CLOCK_MONOTONIC is one clock for every process on a Linux host, so timestamps taken in different
    processes (send time in a worker, arrival time in the coordinator, crash time in the injector) compare."""
    return time.monotonic_ns()


def pin(cpu):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})

# ------------------------------------------------------------------------------------------ jitter models


class Jitter:
    """Extra one-way delay (seconds) added to each frame.

    none     no added delay (loopback plus scheduler noise only)
    expM     exponential delay with mean M milliseconds, e.g. exp5, exp20
    bursty   two-state congestion model in the spirit of Gilbert (1960): a link is normally 'good'
             (exponential delay, mean 2 ms) and enters 'bad' episodes as a Poisson process (rate 0.2/s);
             an episode lasts an exponential time (mean 300 ms) and adds an exponential delay with mean
             100 ms to every frame sent during it.
    """

    def __init__(self, profile, seed):
        self.profile = profile
        self.rng = random.Random(seed)
        self.bad_until = 0.0
        self.next_bad = None

    def sample(self, t):
        p, r = self.profile, self.rng
        if p == "none":
            return 0.0
        if p.startswith("exp"):
            return r.expovariate(1000.0 / float(p[3:]))
        if p == "bursty":
            if self.next_bad is None:
                self.next_bad = t + r.expovariate(0.2)
            while t >= self.next_bad:                       # start the next congestion episode
                self.bad_until = self.next_bad + r.expovariate(1 / 0.300)
                self.next_bad = self.bad_until + r.expovariate(0.2)
            d = r.expovariate(1000.0 / 2.0)
            if t < self.bad_until:
                d += r.expovariate(1000.0 / 100.0)
            return d
        raise ValueError(f"unknown jitter profile {p}")


class JitterSender:
    """Sends frames on one socket after a jitter delay, in FIFO order, from a background thread.
    Keeps per-type counts of messages and bytes for the overhead measurements."""

    def __init__(self, sock, jitter):
        self.sock, self.jitter = sock, jitter
        self.q = deque()
        self.cv = threading.Condition()
        self.last_dep = 0.0
        self.closed = False
        self.count, self.bytes = {}, {}
        threading.Thread(target=self._loop, daemon=True).start()

    def send(self, obj):
        frame = encode(obj)
        k = obj["t"]
        t = time.monotonic()
        with self.cv:
            dep = max(self.last_dep, t + self.jitter.sample(t))
            self.last_dep = dep
            self.q.append((dep, frame))
            self.count[k] = self.count.get(k, 0) + 1
            self.bytes[k] = self.bytes.get(k, 0) + len(frame)
            self.cv.notify()

    def _loop(self):
        while True:
            with self.cv:
                while not self.q and not self.closed:
                    self.cv.wait()
                if self.closed and not self.q:
                    return
                dep, frame = self.q[0]
                wait = dep - time.monotonic()
                if wait > 0:
                    self.cv.wait(wait)
                    continue
                self.q.popleft()
            try:
                self.sock.sendall(frame)
            except OSError:
                return

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify()

# ------------------------------------------------------------------------------------------ the workload

ITER_PER_MS = 9_000       # LCG iterations per millisecond of one E-core of the testbed (calibrated)


def work(n):
    """CPU-bound task: n steps of a linear congruential generator."""
    x = 1
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return x


def work_value(n, seed=1):
    """What work(n) must return, computed in O(log n) by LCG jump-ahead (Knuth 1997, s.3.2.1): the step
    x -> a*x + c is an affine map, so n steps are one affine map built by repeated squaring. The coordinator
    uses it to check every result without redoing the work."""
    m = 0x7FFFFFFF
    A, C = 1, 0
    a, c = 1103515245, 12345
    while n:
        if n & 1:
            A, C = (a * A) & m, (a * C + c) & m
        a, c = (a * a) & m, (a * c + c) & m
        n >>= 1
    return (A * seed + C) & m


def task_iters(tid, mean_ms, cv, seed):
    """Iterations for task tid: mean_ms on average, coefficient of variation cv (gamma distributed)."""
    if cv <= 0:
        return int(mean_ms * ITER_PER_MS)
    r = random.Random(seed * 1_000_003 + tid)
    k = 1 / (cv * cv)
    return max(1, int(r.gammavariate(k, mean_ms / k) * ITER_PER_MS))


def t_quantile(p, df):
    """Student-t quantile by bisection on the regularised incomplete beta (small df, used for 95 % CIs)."""
    from statistics import NormalDist
    if df > 200:
        return NormalDist().inv_cdf(p)

    def cdf(t):
        x = df / (df + t * t)
        return 1 - 0.5 * _ibeta(df / 2, 0.5, x) if t > 0 else 0.5 * _ibeta(df / 2, 0.5, x)
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1 + aa / c; c = c if abs(c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1) < 1e-12:
            break
    return h


def mean_ci(xs):
    """Mean and half-width of the 95 % confidence interval (Student t)."""
    import statistics as st
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), t_quantile(0.975, len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs))


if __name__ == "__main__":
    # Calibration: python3 common.py [cpu]  prints the ITER_PER_MS value for that CPU
    import sys
    pin(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    n = 200_000
    work(n)                                       # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        work(n)
        times.append(time.perf_counter() - t0)
    print(f"{n / (min(times) * 1000):.0f} iterations per ms (ITER_PER_MS is set to {ITER_PER_MS})")
