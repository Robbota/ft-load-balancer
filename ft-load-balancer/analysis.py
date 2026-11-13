"""Metrics and the analytical model, shared by figures.py and the paper builder.

Model (section 4 of the paper). N workers, mean task time s, heartbeat interval h, prefetch depth k,
f workers crash at t_c. delta is the mean one-way delay of a heartbeat, U ~ Uniform(0, h) the time between
the crashed worker's last heartbeat and the crash.

  detection   fixed timeout T:   D = T - U + delta                 E[D] = T - h/2 + delta
              Chen, margin a:    D = h - U + delta + a             E[D] = h/2 + delta + a
              phi accrual:       D = mu - U + delta + z*sigma      E[D] = mu - h/2 + delta + z*sigma
  recovery    the f*k orphaned tasks go to the head of the queue; the j-th is handed out at the j-th
              completion among the N-f survivors and then waits behind k-1 tasks already queued there:
              R = D + f*k*s/(N-f) + k*s
  throughput  pull: X drops at once from N/s to (N-f)/s; detection delay costs no throughput while the
              queue still has work, only the f*k orphans finish late
  makespan    M = max(t_c + (W - C(t_c)) / X_{N-f},  t_c + R)
"""
import bisect
import statistics as st
from statistics import NormalDist

from common import mean_ci

POLL = 0.001                      # mean extra delay from the coordinator's 2 ms detector poll


def window(comp, a, b):
    """Throughput (tasks/s) from completion times in [a, b)."""
    if b <= a:
        return None
    return (bisect.bisect_left(comp, b) - bisect.bisect_left(comp, a)) / (b - a)


def crash_time(r):
    ts = [f["t"] for f in r["faults"] if f["type"] in ("freeze", "kill")]
    return min(ts) if ts else None


def x_pre(r, tc=4.0):
    return window(r["completions"], 1.0, tc)


def x_post(r, tc, gap=0.5, length=3.0):
    end = (r["makespan"] or r["elapsed"]) - 0.3
    return window(r["completions"], tc + gap, min(tc + gap + length, end))


def timeline(r, bin_s=0.1):
    comp = r["completions"]
    T = r["makespan"] or r["elapsed"]
    n = int(T / bin_s) + 1
    xs = [i * bin_s for i in range(n)]
    ys = [window(comp, x, x + bin_s) for x in xs]
    return xs, ys


def detection(runs):
    return [v for r in runs for v in r["detection_s"].values()]


def recovery(runs):
    """Recovery latency of each run: from the crash until the last task stranded on any crashed worker has
    been committed (with f = 1 this is the single worker's value)."""
    out = []
    for r in runs:
        vals = [v for v in r["recovery_s"].values() if v is not None]
        if vals:
            out.append(max(vals))
    return out


def recovery_per_worker(runs):
    return [v for r in runs for v in r["recovery_s"].values() if v is not None]


def stat(xs, scale=1.0):
    m, ci = mean_ci([x * scale for x in xs if x is not None])
    return m, ci


def fmt(xs, scale=1.0, d=1):
    m, ci = stat(xs, scale)
    if m != m:
        return "–"
    return f"{m:.{d}f} ± {ci:.{d}f}"


# ------------------------------------------------------------------------------------------- model

def z_of(phi):
    return NormalDist().inv_cdf(1 - 10.0 ** (-phi))


def model_detection(kind, param, h, delta, mu=None, sigma=None, min_sd=0.005):
    if kind == "fixed":
        return param - h / 2 + delta + POLL
    if kind == "chen":
        return h / 2 + delta + param + POLL
    if kind == "phi":
        mu = h if mu is None else mu
        sigma = max(sigma or 0.0, min_sd)
        return mu - h / 2 + delta + z_of(param) * sigma + POLL
    raise ValueError(kind)


def model_recovery(D, N, f, k, s):
    return D + f * k * s / (N - f) + k * s


def model_makespan(r, N, f, D, k, tc=4.0):
    W = r["cfg"]["tasks"]
    xp = x_pre(r, tc)
    s = N / xp
    done = bisect.bisect_left(r["completions"], tc)
    fluid = tc + (W - done) / (xp * (N - f) / N)
    return max(fluid, tc + model_recovery(D, N, f, k, s))


def hb_stats(traces, h):
    """Mean one-way delay, mean and sd of inter-arrival times (seconds) over recorded traces."""
    d = [(a - s) / 1e9 for tr in traces for _, s, a in tr]
    ia = []
    for tr in traces:
        ia += [(tr[i + 1][2] - tr[i][2]) / 1e9 for i in range(50, len(tr) - 1)]
    return st.mean(d), st.mean(ia), st.pstdev(ia)


def hb_params(e2, jitter="none", h=0.05):
    g = next(x for x in e2 if x["jitter"] == jitter)
    return hb_stats(g["traces"], h)


def annotate(e3, e5a, e2, N=8, h=0.05, e7=None):
    """Adds model_D and model_R (ms) to the experiment groups, using delay statistics from the E2 trace
    recorded without added jitter."""
    delta, mu, sigma = hb_params(e2, "none", h)
    for g in e3 or []:
        if g["det"] in ("fixed", "chen", "phi") and g["f"] > 0:
            D = model_detection(g["det"], g["param"], h, delta, mu, sigma)
            s = st.mean(N / x_pre(r) for r in g["runs"])
            g["model_D"] = D * 1000
            g["model_R"] = model_recovery(D, N, g["f"], g["runs"][0]["cfg"]["k"], s) * 1000
            g["model_M"] = [model_makespan(r, N, g["f"], D, r["cfg"]["k"]) for r in g["runs"]]
    for g in e5a or []:
        D = model_detection("phi", 8.0, h, delta, mu, sigma)
        s = st.mean(N / x_pre(r) for r in g["runs"])
        g["model_D"] = D * 1000
        g["model_R"] = model_recovery(D, N, 1, g["k"], s) * 1000
    for g in e7 or []:
        D = model_detection("phi", 8.0, h, delta, mu, sigma)
        s = st.mean(N / x_pre(r) for r in g["runs"])
        g["model_D"] = D * 1000
        g["model_R"] = model_recovery(D, N, 1, g["k"], s) * 1000
        g["s_meas"] = s * 1000
    return delta, mu, sigma
