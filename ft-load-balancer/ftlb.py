"""Fault-tolerant dynamic load balancer: a coordinator and N worker processes over TCP.

Workers
  Each worker is a separate OS process. It connects to the coordinator, sends HELLO, and then
    - runs the tasks it is given one after another and sends a RESULT for each,
    - sends a heartbeat (HB, with a sequence number and its send time) every h seconds from its own thread.
  Every frame a worker sends goes through a JitterSender, which adds the configured network delay.

Coordinator
  Keeps the task queue and hands work out with a credit (prefetch) window of k tasks per worker:
    pull     one global queue; a worker is topped up to k outstanding tasks whenever a result arrives
             (dynamic load balancing, self-scheduling in the sense of Polychronopoulos and Kuck 1987)
    static   tasks are split into N equal blocks up front; each worker is fed only from its own block
  One failure detector per worker (detectors.py) decides when a worker is suspected. On suspicion the
  coordinator stops giving it work and puts its outstanding tasks (and, for static, the rest of its block)
  back at the front of the queue, from where the survivors take them (task redistribution). If a suspected
  worker later sends a heartbeat, the suspicion was a mistake: the worker is reinstated.
  Results are committed exactly once: the first result for a task wins, later copies are counted as
  duplicate work and dropped, and a CANCEL is sent to any other worker still holding a copy it has not
  started. Every result value is checked against the LCG jump-ahead formula.

Faults (injected from a separate thread that the coordinator logic never reads)
  freeze   SIGSTOP and never resume: a silent host crash. The TCP connection stays open, no FIN is sent,
           so only the heartbeats can reveal it.
  kill     SIGKILL: a process crash on a live host. The kernel closes the socket, and with --eof the
           coordinator treats the end-of-file as a certain failure notice.
  pause    SIGSTOP, then SIGCONT after dur seconds: a long garbage-collection or VM pause. The process has
           not failed, so any suspicion is a detector mistake.
  restart  after a freeze or kill, start a fresh worker process (a new incarnation) on the same CPU.

Examples
  python3 ftlb.py --workers 8 --tasks 2400 --detector phi --param 8
  python3 ftlb.py --workers 8 --tasks 2400 --fault freeze:0:4.0 --fault freeze:1:4.0 --trace
  python3 ftlb.py --workers 8 --tasks 2400 --jitter bursty --detector fixed --param 0.15 --fault pause:3:4:0.4
"""
import argparse
import json
import os
import queue
import resource
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import (Jitter, JitterSender, connect, listen, now_ns, pin, recv, task_iters, work,
                    work_value)
import detectors

HERE = Path(__file__).resolve().parent

# ============================================================================================= worker


def worker_main(a):
    pin(a.cpu)
    sock = connect(a.port)
    out = JitterSender(sock, Jitter(a.jitter, a.seed * 7919 + a.wid))
    out.send({"t": "HELLO", "w": a.wid, "pid": os.getpid()})
    local = deque()                    # (tid, iters, attempt) in arrival order
    cv = threading.Condition()
    state = {"stop": False}

    def reader():
        while True:
            m, _ = recv(sock)
            with cv:
                if m is None or m["t"] == "STOP":
                    state["stop"] = True
                elif m["t"] == "TASK":
                    local.append((m["id"], m["n"], m["a"]))
                elif m["t"] == "CANCEL":
                    for x in list(local):
                        if x[0] == m["id"]:
                            local.remove(x)
                cv.notify()
            if state["stop"]:
                return

    def heartbeats():
        seq, nxt = 0, time.monotonic()
        while not state["stop"]:
            nxt += a.h
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()  # fell behind (e.g. after a pause): send now, no catch-up burst
            seq += 1
            out.send({"t": "HB", "w": a.wid, "s": seq, "ts": now_ns()})

    threading.Thread(target=reader, daemon=True).start()
    if a.h > 0:
        threading.Thread(target=heartbeats, daemon=True).start()
    while True:
        with cv:
            while not local and not state["stop"]:
                cv.wait()
            if state["stop"]:
                break
            tid, n, att = local.popleft()
        t0 = now_ns()
        v = work(n)
        out.send({"t": "RES", "w": a.wid, "id": tid, "v": v, "a": att, "t0": t0, "t1": now_ns()})
    time.sleep(0.05)
    os._exit(0)

# ======================================================================================== coordinator


@dataclass
class Config:
    workers: int = 8
    cpus: list = field(default_factory=lambda: list(range(4, 12)))   # the eight E-cores of the testbed
    coord_cpu: int = 2
    tasks: int = 2400
    task_ms: float = 20.0
    task_cv: float = 0.0
    policy: str = "pull"          # pull | static
    k: int = 4                    # credit window: outstanding tasks per worker
    ft: bool = True               # False: no failure detection and no redistribution
    h: float = 0.05               # heartbeat interval in seconds (0 = no heartbeats)
    detector: str = "phi"         # fixed | chen | phi
    param: float = 8.0            # T (s) for fixed, alpha (s) for chen, phi threshold for phi
    jitter: str = "none"
    faults: list = field(default_factory=list)
    eof: bool = False             # treat a closed connection as a failure notice
    max_idle: float = 3.0         # stop if no task completes for this long (job cannot finish)
    max_time: float = 120.0
    seed: int = 1
    trace: bool = False           # print events
    record_hb: bool = False       # keep every heartbeat arrival (for the offline detector study)


class W:
    def __init__(self, wid, cpu):
        self.wid, self.cpu = wid, cpu
        self.proc = None
        self.out = None
        self.det = None
        self.ready = False
        self.suspected = False
        self.eof = False
        self.assigned = {}            # tid -> attempt
        self.private = deque()        # static policy: this worker's own block
        self.hb = []                  # (seq, send_ns, arrival_ns)


class Coordinator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ev = queue.Queue()
        self.ws = {}
        self.iters = [task_iters(t, cfg.task_ms, cfg.task_cv, cfg.seed) for t in range(cfg.tasks)]
        self.queue = deque()
        self.done = {}                # tid -> (t, wid, attempt)
        self.attempts = [0] * cfg.tasks
        self.holders = {}             # tid -> set of wids that hold it
        self.n_exec = self.n_dup = self.n_wrong = self.n_cancel = 0
        self.suspicions, self.reinstated, self.orphans = [], [], []
        self.fault_log = []
        self.rx_count, self.rx_bytes = {}, {}
        self.t0 = None
        self.next_wid = cfg.workers
        self.lock = threading.Lock()
        self.orphan_of = {}           # tid -> index into self.orphans (instrumentation)
        self.exec_log = []            # (commit time, execution time, worker) of every committed task
        self.qempty_t = None          # time the shared queue first ran empty
        self.qempty_ev = threading.Event()

    # ------------------------------------------------------------------ helpers
    def rel(self, t=None):
        return (time.monotonic() if t is None else t) - self.t0

    def log(self, *a):
        if self.cfg.trace:
            print(f"[{self.rel():8.3f}s]", *a, flush=True)

    def spawn(self, wid, cpu):
        c = self.cfg
        w = W(wid, cpu)
        self.ws[wid] = w
        w.proc = subprocess.Popen([sys.executable, str(HERE / "ftlb.py"), "worker", "--port", str(self.port),
                                   "--wid", str(wid), "--h", str(c.h), "--jitter", c.jitter,
                                   "--seed", str(c.seed), "--cpu", str(cpu)])
        return w

    def eligible(self, w):
        return w.ready and not w.suspected and not w.eof

    def assign(self, w, tid):
        if tid in self.orphan_of:
            rec = self.orphans[self.orphan_of[tid]]
            rec["dispatch"].setdefault(str(tid), (round(self.rel(), 5), w.wid, len(w.assigned)))
        self.attempts[tid] += 1
        w.assigned[tid] = self.attempts[tid]
        self.holders.setdefault(tid, set()).add(w.wid)
        w.out.send({"t": "TASK", "id": tid, "n": self.iters[tid], "a": self.attempts[tid]})

    def fill(self, w):
        if not self.eligible(w):
            return
        src = self.queue if self.cfg.policy == "pull" else w.private
        while len(w.assigned) < self.cfg.k:
            if not src and self.cfg.policy == "static" and self.queue:
                src = self.queue          # redistributed orphans waiting in the shared queue
            if not src:
                return
            tid = src.popleft()
            if not self.queue and self.qempty_t is None and self.t0 is not None and self.cfg.policy == "pull":
                self.qempty_t = self.rel()
                self.qempty_ev.set()
            if tid in self.done or tid in w.assigned:
                continue
            self.assign(w, tid)

    # ------------------------------------------------------------------ network
    def acceptor(self, srv):
        while True:
            try:
                s, _ = srv.accept()
            except OSError:
                return
            s.setsockopt(__import__("socket").IPPROTO_TCP, __import__("socket").TCP_NODELAY, 1)
            threading.Thread(target=self.reader, args=(s,), daemon=True).start()

    def reader(self, s):
        wid = None
        while True:
            m, nb = recv(s)
            t = time.monotonic()
            if m is None:
                self.ev.put(("EOF", wid, None, t))
                return
            k = m["t"]
            with self.lock:
                self.rx_count[k] = self.rx_count.get(k, 0) + 1
                self.rx_bytes[k] = self.rx_bytes.get(k, 0) + nb
            if k == "HELLO":
                wid = m["w"]
                self.ev.put(("HELLO", wid, (m, s), t))
            else:
                self.ev.put((k, wid, m, t))

    # ------------------------------------------------------------------ events
    def on_hello(self, wid, m, s, t):
        c = self.cfg
        w = self.ws[wid]
        w.out = JitterSender(s, Jitter(c.jitter, c.seed * 104729 + wid))
        if c.h > 0:
            w.det = detectors.make(c.detector, c.param, c.h)
            w.det.heartbeat(0, t)
        w.ready = True
        if self.t0 is not None:          # a restarted worker joining a running job
            self.log(f"worker {wid} joined (pid {m['pid']})")
            self.fill(w)

    def on_hb(self, w, m, t):
        if self.cfg.record_hb:
            w.hb.append((m["s"], m["ts"], int(t * 1e9)))
        if w.det is None:
            return
        if w.suspected and not w.eof:
            if hasattr(w.det, "mistake"):
                w.det.mistake()
            w.det.heartbeat(m["s"], t)
            w.suspected = False
            self.reinstated.append({"w": w.wid, "t": self.rel(t)})
            self.log(f"worker {w.wid} reinstated (suspicion was a mistake)")
            self.fill(w)
        else:
            w.det.heartbeat(m["s"], t)

    def on_result(self, w, m, t):
        tid = m["id"]
        self.n_exec += 1
        w.assigned.pop(tid, None)
        self.holders.get(tid, set()).discard(w.wid)
        if tid in self.done:
            self.n_dup += 1
        else:
            if m["v"] != work_value(self.iters[tid]):
                self.n_wrong += 1
            self.done[tid] = (self.rel(t), w.wid, m["a"])
            self.exec_log.append((round(self.rel(t), 4), round((m["t1"] - m["t0"]) / 1e9, 5), w.wid))
            if tid in self.orphan_of:
                rec = self.orphans[self.orphan_of[tid]]
                rec["start"][str(tid)] = round(m["t0"] / 1e9 - self.t0, 5)
                rec["end"][str(tid)] = round(m["t1"] / 1e9 - self.t0, 5)
            for other in list(self.holders.get(tid, ())):     # copies still queued elsewhere
                o = self.ws[other]
                if tid in o.assigned:
                    del o.assigned[tid]
                    self.n_cancel += 1
                    if not o.eof:
                        o.out.send({"t": "CANCEL", "id": tid})
                    self.fill(o)
            self.holders.pop(tid, None)
        self.fill(w)

    def suspect(self, w, t, why):
        w.suspected = True
        self.suspicions.append({"w": w.wid, "t": self.rel(t), "why": why})
        self.log(f"worker {w.wid} suspected ({why})")
        if not self.cfg.ft:
            return
        orphan = [tid for tid in w.assigned if tid not in self.done]
        if self.cfg.policy == "static":
            orphan += [tid for tid in w.private if tid not in self.done]
            w.private.clear()
        for tid in orphan:
            self.holders.get(tid, set()).discard(w.wid)
        w.assigned.clear()
        self.orphans.append({"w": w.wid, "t": self.rel(t), "tids": orphan, "dispatch": {}, "start": {},
                             "end": {}})
        for tid in orphan:
            self.orphan_of[tid] = len(self.orphans) - 1
        if self.cfg.policy == "pull":
            self.queue.extendleft(reversed(orphan))
        else:                           # static: deal the orphans round-robin onto the survivors' blocks
            live = [x for x in self.ws.values() if self.eligible(x)]
            for i, tid in enumerate(reversed(orphan)):
                if live:
                    live[i % len(live)].private.appendleft(tid)
                else:
                    self.queue.appendleft(tid)
        self.log(f"  redistributed {len(orphan)} tasks")
        self.fill_all()

    def fill_all(self):
        """Top up every eligible worker one task at a time in turn, so that requeued orphans are spread
        over the workers with free credit instead of all going to the first one."""
        progress = True
        while progress:
            progress = False
            for x in self.ws.values():
                if self.eligible(x) and len(x.assigned) < self.cfg.k:
                    before = len(x.assigned)
                    k_save = self.cfg.k
                    self.cfg.k = before + 1          # allow exactly one more task
                    self.fill(x)
                    self.cfg.k = k_save
                    progress |= len(x.assigned) > before

    # ------------------------------------------------------------------ fault injection
    def injector(self):
        c = self.cfg
        if any(f["at"] == "qempty" for f in c.faults):   # crash the moment the shared queue runs empty
            self.qempty_ev.wait()
            for f in c.faults:
                if f["at"] == "qempty" and not self.finished:
                    w = self.ws[f["w"]]
                    os.kill(w.proc.pid, signal.SIGKILL if f["type"] == "kill" else signal.SIGSTOP)
                    self.fault_log.append({**f, "t": self.rel()})
                    self.log(f"INJECT {f['type']} worker {f['w']} (queue empty)")
        plan = sorted((f for f in c.faults if f["at"] != "qempty"), key=lambda f: f["at"])
        actions = []
        for f in plan:
            actions.append((f["at"], "start", f))
            if f["type"] == "pause":
                actions.append((f["at"] + f["dur"], "resume", f))
            if f.get("restart") is not None:
                actions.append((f["at"] + f["restart"], "restart", f))
        actions.sort(key=lambda a: a[0])
        for at, what, f in actions:
            d = self.t0 + at - time.monotonic()
            if d > 0:
                time.sleep(d)
            if self.finished:
                return
            w = self.ws[f["w"]]
            if what == "start":
                os.kill(w.proc.pid, signal.SIGKILL if f["type"] == "kill" else signal.SIGSTOP)
                self.fault_log.append({**f, "t": self.rel()})
                self.log(f"INJECT {f['type']} worker {f['w']}")
            elif what == "resume":
                os.kill(w.proc.pid, signal.SIGCONT)
                self.fault_log.append({"type": "resume", "w": f["w"], "t": self.rel()})
                self.log(f"INJECT resume worker {f['w']}")
            else:
                with self.lock:
                    wid = self.next_wid
                    self.next_wid += 1
                self.ev.put(("SPAWN", wid, w.cpu, time.monotonic()))
                self.fault_log.append({"type": "restart", "w": f["w"], "new": wid, "t": self.rel()})

    # ------------------------------------------------------------------ main loop
    def run(self):
        c = self.cfg
        pin(c.coord_cpu)
        srv = listen()
        self.port = srv.getsockname()[1]
        threading.Thread(target=self.acceptor, args=(srv,), daemon=True).start()
        for i in range(c.workers):
            self.spawn(i, c.cpus[i % len(c.cpus)] if c.cpus else None)
        ready = 0
        while ready < c.workers:
            kind, wid, m, t = self.ev.get()
            if kind == "HELLO":
                self.on_hello(wid, m[0], m[1], t)
                ready += 1
        # initial distribution
        tids = list(range(c.tasks))
        if c.policy == "pull":
            self.queue.extend(tids)
        else:
            per = -(-c.tasks // c.workers)
            for i in range(c.workers):
                self.ws[i].private.extend(tids[i * per:(i + 1) * per])
        self.finished = False
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self.t0 = time.monotonic()
        for w in self.ws.values():
            # a fresh detector anchored at t0: start-up time must not count as silence, and the gap between
            # HELLO and t0 must not enter the detector's statistics as if it were a heartbeat interval
            if w.det is not None:
                w.det = detectors.make(c.detector, c.param, c.h)
                w.det.heartbeat(0, self.t0)
            self.fill(w)
        threading.Thread(target=self.injector, daemon=True).start()
        last_done = self.t0
        end_reason = "complete"
        while len(self.done) < c.tasks:
            try:
                kind, wid, m, t = self.ev.get(timeout=0.002)
            except queue.Empty:
                kind = None
            if kind == "HB":
                self.on_hb(self.ws[wid], m, t)
            elif kind == "RES":
                self.on_result(self.ws[wid], m, t)
                last_done = t
            elif kind == "HELLO":
                self.on_hello(wid, m[0], m[1], t)
            elif kind == "SPAWN":
                self.spawn(wid, m)
            elif kind == "EOF" and wid is not None:
                w = self.ws[wid]
                w.eof = True
                if c.eof and not w.suspected:
                    self.suspect(w, t, "eof")
            now = time.monotonic()
            if c.ft and c.h > 0:
                for w in self.ws.values():
                    if w.ready and not w.suspected and w.det is not None and now > w.det.deadline():
                        self.suspect(w, now, "timeout")
            if now - last_done > c.max_idle:
                end_reason = "stalled"
                break
            if now - self.t0 > c.max_time:
                end_reason = "timeout"
                break
        t_end = time.monotonic()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        self.finished = True
        # shut down
        for w in self.ws.values():
            if w.proc and w.proc.poll() is None:
                try:
                    os.kill(w.proc.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                if w.out and not w.eof:
                    w.out.send({"t": "STOP"})
        time.sleep(0.15)
        for w in self.ws.values():
            if w.proc and w.proc.poll() is None:
                w.proc.kill()
            if w.proc:
                w.proc.wait()
        srv.close()
        return self.summary(t_end, end_reason, ru0, ru1)

    # ------------------------------------------------------------------ results
    def summary(self, t_end, end_reason, ru0, ru1):
        c = self.cfg
        tx_count, tx_bytes = {}, {}
        for w in self.ws.values():
            if w.out:
                for k, v in w.out.count.items():
                    tx_count[k] = tx_count.get(k, 0) + v
                for k, v in w.out.bytes.items():
                    tx_bytes[k] = tx_bytes.get(k, 0) + v
        crashes = {f["w"]: f["t"] for f in self.fault_log if f["type"] in ("freeze", "kill")}
        pauses = [(f["w"], f["t"]) for f in self.fault_log if f["type"] == "pause"]
        # classify suspicions: correct only if the worker had crashed for good before the suspicion
        det_lat, mistakes = {}, []
        for s in self.suspicions:
            if s["w"] in crashes and s["t"] >= crashes[s["w"]]:
                det_lat.setdefault(s["w"], s["t"] - crashes[s["w"]])
            else:
                mistakes.append(s)
        rec_lat = {}
        for o in self.orphans:
            if o["w"] in crashes:
                ts = [self.done[t][0] for t in o["tids"] if t in self.done]
                ok = all(t in self.done for t in o["tids"])
                rec_lat[o["w"]] = ((max(ts) if ts else o["t"]) - crashes[o["w"]]) if ok else None
        out = {
            "cfg": asdict(c),
            "end": end_reason,
            "makespan": t_end - self.t0 if end_reason == "complete" else None,
            "elapsed": t_end - self.t0,
            "completed": len(self.done),
            "executions": self.n_exec,
            "duplicates": self.n_dup,
            "cancelled": self.n_cancel,
            "wrong": self.n_wrong,
            "commits_ok": len(self.done) == c.tasks and self.n_wrong == 0,
            "completions": sorted(round(v[0], 4) for v in self.done.values()),
            "faults": self.fault_log,
            "suspicions": self.suspicions,
            "reinstated": self.reinstated,
            "mistakes": mistakes,
            "detection_s": {str(k): v for k, v in det_lat.items()},
            "recovery_s": {str(k): v for k, v in rec_lat.items()},
            "orphaned": {str(o["w"]): len(o["tids"]) for o in self.orphans},
            "orphan_detail": self.orphans,
            "exec_log": self.exec_log,
            "qempty_t": self.qempty_t,
            "tx_count": tx_count, "tx_bytes": tx_bytes,
            "rx_count": self.rx_count, "rx_bytes": self.rx_bytes,
            "coord_cpu_s": (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime),
            "pauses": pauses,
        }
        if c.record_hb:
            out["hb"] = {str(w.wid): w.hb for w in self.ws.values()}
        return out


def run(cfg):
    return Coordinator(cfg).run()


def parse_fault(s):
    """type:worker:at[:dur_or_restart]  e.g. freeze:0:4.0   pause:3:4:0.4   freeze:1:4:r2.0 (restart after 2 s)
    at may be 'qempty': inject the moment the shared queue first runs empty (pull policy)."""
    p = s.split(":")
    f = {"type": p[0], "w": int(p[1]), "at": p[2] if p[2] == "qempty" else float(p[2])}
    if len(p) > 3:
        if p[3].startswith("r"):
            f["restart"] = float(p[3][1:])
        else:
            f["dur"] = float(p[3])
    return f


def report(r):
    c = r["cfg"]
    print(f"policy={c['policy']} k={c['k']} ft={c['ft']} detector={c['detector']}({c['param']}) "
          f"h={c['h']*1000:.0f} ms jitter={c['jitter']} workers={c['workers']} tasks={c['tasks']}")
    if r["makespan"] is not None:
        print(f"  makespan {r['makespan']:.3f} s, throughput {r['completed']/r['makespan']:.1f} tasks/s")
    else:
        print(f"  job did NOT finish ({r['end']}): {r['completed']}/{c['tasks']} tasks committed "
              f"after {r['elapsed']:.2f} s")
    print(f"  executions {r['executions']}, duplicates {r['duplicates']}, cancelled copies {r['cancelled']}, "
          f"wrong results {r['wrong']}, every task committed once: {r['commits_ok']}")
    for f in r["faults"]:
        print(f"  fault  {f['type']:7s} worker {f['w']} at {f['t']:.3f} s")
    for s in r["suspicions"]:
        print(f"  suspect worker {s['w']} at {s['t']:.3f} s ({s['why']})")
    for k, v in r["detection_s"].items():
        rec = r["recovery_s"].get(k)
        print(f"  worker {k}: detection {v*1000:.1f} ms, orphaned {r['orphaned'].get(k, 0)} tasks, "
              f"recovery {'n/a' if rec is None else f'{rec*1000:.1f} ms'}")
    if r["mistakes"]:
        print(f"  mistaken suspicions: {len(r['mistakes'])}, reinstated: {len(r['reinstated'])}")
    hb = r["tx_count"].get("HB", 0) + r["rx_count"].get("HB", 0)
    print(f"  heartbeats received {r['rx_count'].get('HB', 0)}, coordinator CPU {r['coord_cpu_s']:.2f} s")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        ap = argparse.ArgumentParser()
        ap.add_argument("worker")
        for a, t in (("--port", int), ("--wid", int), ("--h", float), ("--jitter", str), ("--seed", int),
                     ("--cpu", int)):
            ap.add_argument(a, type=t)
        worker_main(ap.parse_args())
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tasks", type=int, default=2400)
    ap.add_argument("--task-ms", type=float, default=20.0)
    ap.add_argument("--task-cv", type=float, default=0.0)
    ap.add_argument("--policy", default="pull", choices=["pull", "static"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--no-ft", action="store_true")
    ap.add_argument("--h", type=float, default=0.05)
    ap.add_argument("--detector", default="phi", choices=["fixed", "chen", "chen-r", "phi"])
    ap.add_argument("--param", type=float, default=8.0)
    ap.add_argument("--jitter", default="none")
    ap.add_argument("--fault", action="append", default=[], help="type:worker:at[:dur|:rS]")
    ap.add_argument("--eof", action="store_true")
    ap.add_argument("--cpus", default="4-11", help="CPU list for workers, e.g. 4-11 or 'none'")
    ap.add_argument("--coord-cpu", default="2")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", help="write the full result to this file")
    a = ap.parse_args()
    if a.cpus == "none":
        cpus = []
    else:
        lo, _, hi = a.cpus.partition("-")
        cpus = list(range(int(lo), int(hi or lo) + 1))
    cfg = Config(workers=a.workers, tasks=a.tasks, task_ms=a.task_ms, task_cv=a.task_cv, policy=a.policy,
                 k=a.k, ft=not a.no_ft, h=a.h, detector=a.detector, param=a.param, jitter=a.jitter,
                 faults=[parse_fault(f) for f in a.fault], eof=a.eof, cpus=cpus,
                 coord_cpu=None if a.coord_cpu == "none" else int(a.coord_cpu), seed=a.seed, trace=a.trace)
    r = run(cfg)
    report(r)
    if a.json:
        Path(a.json).write_text(json.dumps(r))


if __name__ == "__main__":
    main()
