"""Offline quality-of-service evaluation of the failure detectors on recorded heartbeat traces.

The metrics are those of Chen, Toueg and Aguilera (2002):
  T_D   detection time: from a crash to the moment the detector suspects the crashed process for good
  lambda_M  mistake rate: wrong suspicions per unit time when nothing has failed
  T_M   mistake duration: how long a wrong suspicion lasts until the next heartbeat corrects it
  P_A   query accuracy probability: fraction of time the detector's output is correct

Each trace is a list of (seq, send_ns, arrival_ns) for one worker, recorded by ftlb.py during a fault-free
run (both timestamps come from the same CLOCK_MONOTONIC). Replaying the arrivals through a detector gives
its deadline after every arrival; a mistake is an arrival later than the previous deadline. For T_D, crash
times are drawn uniformly over the trace: a crash at time t stops every heartbeat sent after t, so the
detector's last state is the one after the last heartbeat sent before t, and T_D = deadline - t.
This trace-driven method is the one used by Hayashibara et al. (2004) and Satzger et al. (2007).
"""
import bisect
import random
import statistics as st

import detectors

WARMUP = 50          # arrivals used only to fill the detectors' windows


def replay(trace, kind, param, h):
    """Deadline after every arrival. As in the live coordinator, an arrival after the current deadline ends
    a wrong suspicion, and the detector is told about the mistake (only 'chen-r' acts on it)."""
    det = detectors.make(kind, param, h)
    dls = []
    for i, (seq, _, arr) in enumerate(trace):
        t = arr / 1e9
        if i > 0 and t > dls[-1] and hasattr(det, "mistake"):
            det.mistake()
        det.heartbeat(seq, t)
        dls.append(det.deadline())
    return dls


def evaluate(traces, kind, param, h, crashes_per_trace=400, seed=1):
    rng = random.Random(seed)
    mistakes, durations, tds, span = 0, [], [], 0.0
    for trace in traces:
        dls = replay(trace, kind, param, h)
        arr = [a / 1e9 for _, _, a in trace]
        snd = [s / 1e9 for _, s, _ in trace]
        for i in range(WARMUP, len(trace) - 1):
            if arr[i + 1] > dls[i]:
                mistakes += 1
                durations.append(arr[i + 1] - dls[i])
        span += arr[-1] - arr[WARMUP]
        lo, hi = snd[WARMUP], snd[-1] - 2 * h
        for _ in range(crashes_per_trace):
            t = rng.uniform(lo, hi)
            j = bisect.bisect_left(snd, t) - 1         # last heartbeat sent before the crash
            tds.append(max(0.0, dls[j] - t))
    tds.sort()
    return {
        "kind": kind, "param": param,
        "td_mean": st.mean(tds), "td_p50": tds[len(tds) // 2], "td_p99": tds[int(0.99 * len(tds))],
        "td_max": tds[-1],
        "mistakes": mistakes, "span_s": span,
        "lambda_m_per_min": 60 * mistakes / span,
        "tm_mean": st.mean(durations) if durations else 0.0,
        "pa": 1 - sum(durations) / span,
    }


SWEEP = {
    "fixed": [0.055, 0.06, 0.065, 0.075, 0.09, 0.11, 0.15, 0.2, 0.3, 0.45, 0.7],
    "chen": [0.001, 0.003, 0.006, 0.01, 0.02, 0.035, 0.06, 0.1, 0.18, 0.3, 0.5],
    "phi": [0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12],
}
SWEEP["chen-r"] = SWEEP["chen"]       # Chen with the estimator reset after each mistake


def delays_ms(traces):
    return [(a - s) / 1e6 for tr in traces for _, s, a in tr]


def intervals_ms(traces):
    out = []
    for tr in traces:
        out += [(tr[i + 1][2] - tr[i][2]) / 1e6 for i in range(len(tr) - 1)]
    return out
