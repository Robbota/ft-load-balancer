"""Runs every experiment of the term paper and writes results/<name>.json.

  python3 run_experiments.py              # all experiments, about 100 minutes on the testbed
  python3 run_experiments.py --only E3 E4 # selected experiments
  python3 run_experiments.py --reps 2     # fewer repetitions (quick check; wider confidence intervals)

Within each experiment the settings are interleaved: repetition 1 of every setting runs first, then
repetition 2 of every setting, and so on, so that slow drift in the machine (the CPU clock falling as the
laptop heats up) affects all settings alike (Georges, Buytaert and Eeckhout 2007).

E1  heartbeat overhead: no faults, heartbeat interval h from 5 ms to 250 ms (and none)
E2  detector QoS: record heartbeat traces under four jitter profiles; offline sweep of the detectors
E3  crash recovery: f = 0, 1, 2, 4 of 8 workers frozen at t = 4 s, three detectors; process crash with
    end-of-file notice; crash followed by restart
E4  jitter and pauses: four jitter profiles x four detectors (incl. Chen reset), one worker paused 0.3 s
E5a prefetch depth k = 1..16 with one crash
E5  redistribution: static vs pull with and without fault tolerance, no crash / early / late crash,
    variable task sizes; heterogeneous cores
E6  end-of-job crash: one worker frozen the moment the shared queue runs empty, three detectors
E7  task size: s = 2, 20, 80 ms x prefetch k = 1, 4, one worker frozen at 4 s
"""
import argparse
import json
import time
from pathlib import Path

import ftlb
import qos
from ftlb import Config

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)

N = 8
TASKS = 4000                      # about 11 s with 8 workers
DETECTORS = [("fixed", 0.15), ("chen", 0.05), ("phi", 8.0)]
DETECTORS4 = DETECTORS[:2] + [("chen-r", 0.05)] + DETECTORS[2:]
JITTERS = ["none", "exp5", "exp20", "bursty"]


def save(name, obj):
    (RES / f"{name}.json").write_text(json.dumps(obj))
    print(f"   -> results/{name}.json", flush=True)


def one(label, **kw):
    t = time.time()
    r = ftlb.run(Config(**kw))
    if r["makespan"]:
        ms = f"{r['makespan']:.2f} s"
    elif r["end"] == "timeout":
        ms = f"stopped at {r['elapsed']:.0f} s ({r['completed']} tasks)"
    else:
        ms = f"NOT finished ({r['completed']} tasks)"
    det = ", ".join(f"D={v*1000:.0f}ms" for v in r["detection_s"].values())
    mis = f", {len(r['mistakes'])} mistakes" if r["mistakes"] else ""
    print(f"  {label:44s} {ms:>26s}  dup={r['duplicates']:<3d}{det and ' ' + det}{mis}  "
          f"[{time.time()-t:.0f}s]", flush=True)
    assert r["wrong"] == 0
    return r


def slim(r, keep_exec=False):
    r = dict(r)
    r.pop("hb", None)
    if not keep_exec:
        r.pop("exec_log", None)
    return r


def sweep(settings, reps, keep_exec=False):
    """settings: list of (key dict, label, Config kwargs). Runs reps repetitions, interleaved, and returns
    [{**key, "runs": [...]}, ...] in the order given."""
    runs = [[] for _ in settings]
    for i in range(reps):
        for j, (key, label, kw) in enumerate(settings):
            runs[j].append(slim(one(f"{label} rep {i}", seed=i + 1, **kw), keep_exec))
    return [{**key, "runs": r} for (key, _, _), r in zip(settings, runs)]


def e1(reps):
    print("E1 heartbeat overhead")
    hs = (0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
    save("e1_overhead", sweep([({"h": h}, f"h={h*1000:.0f}ms", dict(h=h, tasks=2400)) for h in hs], reps))


def e2(reps):
    print("E2 heartbeat traces and offline detector QoS")
    out = []
    for j in JITTERS:
        r = one(f"trace jitter={j}", jitter=j, tasks=N * 45 * 60, ft=False, record_hb=True, max_time=70,
                seed=11)
        traces = [r["hb"][w] for w in sorted(r["hb"])]
        sweep = []
        for kind, params in qos.SWEEP.items():
            for p in params:
                sweep.append(qos.evaluate(traces, kind, p, 0.05))
        out.append({"jitter": j, "traces": traces, "sweep": sweep,
                    "elapsed": r["elapsed"], "completed": r["completed"]})
        best = {k: min((s for s in sweep if s["kind"] == k and s["mistakes"] == 0),
                       key=lambda s: s["td_mean"], default=None) for k in qos.SWEEP}
        for k, s in best.items():
            print(f"     {k:6s} fastest with no mistakes: " +
                  (f"param {s['param']}, T_D mean {s['td_mean']*1000:.1f} ms" if s else "none"))
    save("e2_qos", out)


def freeze(ws, at, **extra):
    return [{"type": "freeze", "w": w, "at": at, **extra} for w in ws]


def e3(reps):
    print("E3 crash recovery")
    st = [({"f": 0, "det": "phi"}, "f=0", dict(tasks=TASKS))]
    for kind, p in DETECTORS:
        for f in (1, 2, 4):
            st.append(({"f": f, "det": kind, "param": p}, f"{kind}({p}) f={f}",
                       dict(tasks=TASKS, detector=kind, param=p, faults=freeze(range(f), 4.0))))
    st.append(({"f": 1, "det": "eof"}, "kill+eof f=1",
               dict(tasks=TASKS, eof=True, faults=[{"type": "kill", "w": 0, "at": 4.0}])))
    st.append(({"f": 2, "det": "phi-restart"}, "freeze+restart f=2",
               dict(tasks=TASKS, faults=freeze((0, 1), 4.0, restart=2.0))))
    save("e3_crash", sweep(st, reps, keep_exec=True))


def e4(reps):
    print("E4 jitter and a 0.3 s pause")
    st = [({"jitter": j, "det": k, "param": p}, f"{j} {k}({p})",
           dict(jitter=j, detector=k, param=p, tasks=TASKS,
                faults=[{"type": "pause", "w": 3, "at": 4.0, "dur": 0.3}]))
          for j in JITTERS for k, p in DETECTORS4]
    save("e4_jitter", sweep(st, reps))


def e5a(reps):
    print("E5a prefetch depth")
    st = [({"k": k}, f"pull k={k} crash", dict(k=k, tasks=TASKS, faults=freeze([0], 4.0)))
          for k in (1, 2, 4, 8, 16)]
    save("e5a_prefetch", sweep(st, reps))


def e5(reps):
    print("E5 redistribution policies")
    st = []
    for pol, ft in (("static", False), ("static", True), ("pull", True)):
        for when in (None, 2.0, 10.0):
            st.append(({"policy": pol, "ft": ft, "crash_at": when}, f"{pol} ft={ft} crash@{when}",
                       dict(policy=pol, ft=ft, tasks=TASKS, task_cv=0.5,
                            faults=[] if when is None else freeze([0], when))))
    cpus = [4, 5, 6, 7, 8, 9, 12, 13]                      # six E-cores and the two low-power E-cores
    ht = [({"policy": pol, "crash_at": when}, f"hetero {pol} crash@{when}",
           dict(policy=pol, cpus=cpus, tasks=TASKS, faults=[] if when is None else freeze([0], when)))
          for pol in ("static", "pull") for when in (None, 4.0)]
    res = sweep(st + ht, reps)
    save("e5_policy", {"policy": res[:len(st)], "hetero": res[len(st):]})


def e6(reps):
    print("E6 crash when the queue runs empty")
    st = [({"det": "none", "crash": False}, "no crash", dict(tasks=TASKS))]
    st += [({"det": k, "param": p, "crash": True}, f"{k}({p}) crash@qempty",
            dict(tasks=TASKS, detector=k, param=p, faults=freeze([0], "qempty")))
           for k, p in DETECTORS]
    save("e6_endcrash", sweep(st, reps))


def e7(reps):
    print("E7 task size")
    st = []
    for s_ms in (2, 20, 80):
        tasks = int(N * 12_000 / (s_ms * 1.35))            # about 12 s of work at the sustained clock
        for k in (1, 4):
            st.append(({"s_ms": s_ms, "k": k}, f"s={s_ms}ms k={k} crash",
                       dict(task_ms=s_ms, k=k, tasks=tasks, faults=freeze([0], 4.0))))
    save("e7_tasksize", sweep(st, reps, keep_exec=True))


def main():
    ap = argparse.ArgumentParser()
    names = ["E1", "E2", "E3", "E4", "E5a", "E5", "E6", "E7"]
    ap.add_argument("--only", nargs="*", default=names)
    ap.add_argument("--reps", type=int, default=None)
    a = ap.parse_args()
    t = time.time()
    for name, fn in (("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4), ("E5a", e5a), ("E5", e5), ("E6", e6),
                     ("E7", e7)):
        if name in a.only:
            fn(a.reps or (1 if name == "E2" else 5))
    print(f"done in {(time.time() - t) / 60:.1f} min")


if __name__ == "__main__":
    main()
