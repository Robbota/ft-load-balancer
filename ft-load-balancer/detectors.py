"""Heartbeat failure detectors.

Each detector watches one monitored process. heartbeat(seq, t) is called when heartbeat number seq arrives
at time t (seconds); deadline() is the time at which the detector will suspect the process if no further
heartbeat arrives. All three detectors here are monotone in the time since the last arrival, so deadline()
describes them completely: the online coordinator suspects when now > deadline(), and the offline QoS
evaluation (qos.py) replays recorded arrival traces through the same objects.

FixedTimeout   suspect when no heartbeat for T seconds (the classic timeout detector).
Chen           Chen, Toueg and Aguilera (2002): estimate the expected arrival time of the next heartbeat
               from the last n arrivals and add a constant safety margin alpha.
PhiAccrual     Hayashibara et al. (2004): fit a normal distribution to the last n inter-arrival times and
               output phi = -log10(P(next heartbeat later than now)); suspect when phi exceeds a threshold.
               The deadline is last + mean + z * sd with z the normal quantile for 1 - 10^-phi.
"""
from collections import deque
from math import sqrt
from statistics import NormalDist


class FixedTimeout:
    name = "fixed"

    def __init__(self, T, h=None):
        self.T = T
        self.last = None

    def heartbeat(self, seq, t):
        self.last = t

    def deadline(self):
        return self.last + self.T


class Chen:
    name = "chen"

    def __init__(self, alpha, h, n=100, reset=False):
        self.alpha, self.h, self.n, self.reset = alpha, h, n, reset
        self.win = deque()            # A_i - h * seq_i
        self.sum = 0.0
        self.seq = 0

    def heartbeat(self, seq, t):
        if seq <= self.seq and self.win:
            return                    # an older heartbeat overtaken by a newer one adds no information
        v = t - self.h * seq
        self.win.append(v)
        self.sum += v
        if len(self.win) > self.n:
            self.sum -= self.win.popleft()
        self.seq = seq

    def deadline(self):
        ea = self.sum / len(self.win) + self.h * (self.seq + 1)
        return ea + self.alpha

    def mistake(self):
        """Called when a suspicion turns out to be wrong. A sender that was paused resumes with its
        heartbeats shifted in phase, so A_i - h*s_i jumps; the old window then predicts arrivals too early
        until it has been flushed. With reset=True the window restarts from the next heartbeat."""
        if self.reset:
            self.win.clear()
            self.sum = 0.0


class PhiAccrual:
    name = "phi"

    def __init__(self, phi, h, n=200, min_sd=0.005):
        self.phi, self.h, self.n, self.min_sd = phi, h, n, min_sd
        self.z = NormalDist().inv_cdf(1 - 10.0 ** (-phi))
        self.win = deque()
        self.s1 = self.s2 = 0.0
        self.last = None

    def heartbeat(self, seq, t):
        if self.last is not None:
            if t <= self.last:
                return
            x = t - self.last
            self.win.append(x)
            self.s1 += x
            self.s2 += x * x
            if len(self.win) > self.n:
                y = self.win.popleft()
                self.s1 -= y
                self.s2 -= y * y
        self.last = t

    def stats(self):
        k = len(self.win)
        if k < 2:                     # bootstrap until two intervals have been seen
            return self.h, self.h / 4
        m = self.s1 / k
        var = max(0.0, self.s2 / k - m * m)
        return m, max(sqrt(var), self.min_sd)

    def deadline(self):
        m, sd = self.stats()
        return self.last + m + self.z * sd


def make(kind, param, h):
    if kind == "chen-r":
        return Chen(param, h, reset=True)
    return {"fixed": FixedTimeout, "chen": Chen, "phi": PhiAccrual}[kind](param, h)
