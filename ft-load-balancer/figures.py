"""Draws every figure of the term paper from results/*.json into figures/.

  python3 figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import analysis as A
import qos

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
C = {"fixed": "#1f77b4", "chen": "#2ca02c", "chen-r": "#98df8a", "phi": "#d62728", "eof": "#9467bd",
     "none": "#444444", "exp5": "#1f77b4", "exp20": "#ff7f0e", "bursty": "#d62728"}
NAMES = {"fixed": "Fixed timeout", "chen": "Chen et al.", "chen-r": "Chen, reset", "phi": "φ accrual"}


CACHE = {}


def load(n):
    if CACHE.get(n) is not None:
        return CACHE[n]
    p = RES / f"{n}.json"
    return json.loads(p.read_text()) if p.exists() else None


def architecture():
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    ax.set_xlim(0, 104); ax.set_ylim(0, 52); ax.axis("off"); ax.grid(False)

    def box(x, y, w, h, text, fc, fs=7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4", fc=fc, ec="#333", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(a, b, color="#333", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=0.9, ls=ls))

    box(6, 7, 32, 38, "", "#eef3fa")
    ax.text(22, 42, "Coordinator (CPU 2)", ha="center", fontsize=8.5, weight="bold")
    box(9, 30, 26, 8, "task queue (pull) or\nper-worker blocks (static)", "white")
    box(9, 19, 26, 8, "one failure detector per\nworker: fixed | Chen | φ", "white")
    box(9, 8.5, 26, 8, "commit table: first result\nwins; other copies cancelled", "white")
    arrow((7.5, 23), (7.5, 34), "#555")
    ax.text(3.6, 28.5, "requeue orphans\non suspicion", fontsize=6, rotation=90, va="center", ha="center")
    ys = (38, 27, 16, 3)
    for i, y in enumerate(ys):
        lab = "worker 7" if i == 3 else f"worker {i}"
        box(62, y, 24, 8, f"{lab}\ncompute + heartbeats", "#fde2e2" if i == 1 else "#f6f6f6")
    ax.text(74, 12.6, "· · ·", fontsize=9, rotation=90, ha="center", va="center")
    ax.text(74, 48, "8 worker processes, CPUs 4–11 (E-cores)", ha="center", fontsize=7)
    ax.text(74, 36.3, "frozen (SIGSTOP)", fontsize=6.5, color="#b00000", ha="center")
    arrow((38.5, 40), (61.5, 43))
    ax.text(49, 43.5, "TASK (≤ k outstanding)", fontsize=6.5, ha="center")
    arrow((61.5, 40), (38.5, 36))
    ax.text(50, 36.0, "RESULT", fontsize=6.5, ha="center")
    arrow((61.5, 29), (38.5, 23), "#a05a00")
    ax.text(50, 23.6, "HB every h", fontsize=6.5, ha="center", color="#a05a00")
    ax.text(51.5, 1.0, "every link adds the jitter delay\n(FIFO, head-of-line blocking)", fontsize=6.3,
            ha="center", color="#a05a00")
    box(91, 20, 11, 15, "fault\ninjector\n\nSIGSTOP\nSIGKILL\nSIGCONT\nrestart", "#fff5e0", 6.3)
    arrow((90.5, 30), (86.5, 31), "#b00000")
    fig.savefig(FIG / "f1_architecture.png")
    plt.close(fig)


def e1():
    d = load("e1_overhead")
    if not d:
        return
    hs = [x["h"] for x in d if x["h"] > 0]
    base = [r["completed"] / r["makespan"] for r in d[0]["runs"]]
    b = sum(base) / len(base)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    thr = [A.stat([r["completed"] / r["makespan"] / b * 100 for r in x["runs"]]) for x in d if x["h"] > 0]
    a1.errorbar([h * 1000 for h in hs], [m for m, _ in thr], yerr=[c for _, c in thr], marker="o", ms=4,
                capsize=3, color="#1f77b4")
    a1.axhline(100, ls="--", color="gray", lw=0.8)
    a1.set_xscale("log"); a1.set_xlabel("heartbeat interval h (ms)"); a1.set_ylabel("throughput (% of no heartbeats)")
    a1.set_title("(a) Failure-free throughput", fontsize=9)
    cpu = [A.stat([r["coord_cpu_s"] / r["makespan"] * 100 for r in x["runs"]]) for x in d]
    xs = [max(x["h"], 0.001) * 1000 for x in d]
    hb = [A.stat([r["rx_count"].get("HB", 0) / r["makespan"] for r in x["runs"]]) for x in d]
    a2.errorbar([h * 1000 for h in hs], [m for m, _ in cpu[1:]], yerr=[c for _, c in cpu[1:]], marker="s",
                ms=4, capsize=3, color="#d62728", label="coordinator CPU (%)")
    a2.axhline(cpu[0][0], ls="--", color="#d62728", lw=0.8, label="CPU without heartbeats")
    a2.set_xscale("log"); a2.set_xlabel("heartbeat interval h (ms)"); a2.set_ylabel("coordinator CPU (% of one core)")
    a3 = a2.twinx()
    a3.plot([h * 1000 for h in hs], [m for m, _ in hb[1:]], "^:", ms=4, color="#555", label="heartbeats/s")
    a3.set_yscale("log"); a3.set_ylabel("heartbeats received per s"); a3.grid(False)
    a2.set_title("(b) Coordinator cost", fontsize=9)
    h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "f2_overhead.png")
    plt.close(fig)


def e2():
    d = load("e2_qos")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for x in d:
        dl = sorted(qos.delays_ms(x["traces"]))
        n = len(dl)
        ax.plot(dl, [(i + 1) / n for i in range(n)], label=x["jitter"], color=C[x["jitter"]])
    ax.set_xscale("log"); ax.set_xlabel("one-way heartbeat delay (ms, log scale)"); ax.set_ylabel("CDF")
    ax.legend(title="jitter profile", fontsize=7, title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "f3_delay_cdf.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True)
    for ax, x in zip(axs, d):
        for kind in ("fixed", "chen", "chen-r", "phi"):
            pts = [s for s in x["sweep"] if s["kind"] == kind]
            if not pts:
                continue
            ax.plot([s["td_mean"] * 1000 for s in pts], [max(s["lambda_m_per_min"], 0.01) for s in pts],
                    "o-", ms=3, lw=1, color=C[kind], label=NAMES[kind])
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(x["jitter"], fontsize=9)
        ax.set_xlabel("mean $T_D$ (ms)")
        ax.axhline(0.01, color="gray", lw=0.6, ls=":")
    axs[0].set_ylabel("mistakes per minute\n(0 drawn at 0.01)")
    axs[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "f4_qos.png"); plt.close(fig)


def e3():
    d = load("e3_crash")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    pick = [("f=0", [g for g in d if g["f"] == 0]), ]
    for f in (1, 2, 4):
        pick.append((f"f={f} (φ accrual)", [g for g in d if g["f"] == f and g["det"] == "phi"]))
    pick.append(("f=2, restarted after 2 s", [g for g in d if g["det"] == "phi-restart"]))
    cols = ["#444", "#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    for (lab, gs), col in zip(pick, cols):
        r = gs[0]["runs"][0]
        xs, ys = A.timeline(r, 0.25)
        ax.step(xs, ys, where="post", label=lab, color=col, lw=1.1)
    ax.axvline(4.0, color="k", ls=":", lw=0.8)
    ax.text(4.05, 20, "crash", fontsize=7)
    ax.axvline(6.0, color="#2ca02c", ls=":", lw=0.8)
    ax.text(6.05, 20, "restart", fontsize=7, color="#2ca02c")
    ax.set_xlabel("time since job start (s)"); ax.set_ylabel("throughput (tasks/s, 250 ms bins)")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(FIG / "f5_timeline.png"); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    width = 0.25
    for j, kind in enumerate(("fixed", "chen", "phi")):
        gs = [g for g in d if g["det"] == kind and g["f"] > 0]
        fs = [g["f"] for g in gs]
        dm = [A.stat(A.detection(g["runs"]), 1000) for g in gs]
        rm = [A.stat(A.recovery(g["runs"]), 1000) for g in gs]
        pos = [i + (j - 1) * width for i in range(len(fs))]
        a1.bar(pos, [m for m, _ in dm], width, yerr=[c for _, c in dm], capsize=2, color=C[kind], label=NAMES[kind])
        a2.bar(pos, [m for m, _ in rm], width, yerr=[c for _, c in rm], capsize=2, color=C[kind], label=NAMES[kind])
        a2.scatter(pos, [g.get("model_R", float("nan")) for g in gs], marker="_", s=120, color="k", zorder=5,
                   label="model" if j == 0 else None)
        a1.scatter(pos, [g.get("model_D", float("nan")) for g in gs], marker="_", s=120, color="k", zorder=5,
                   label="model" if j == 0 else None)
    for a, t in ((a1, "(a) Detection latency"), (a2, "(b) Recovery latency")):
        a.set_xticks(range(3)); a.set_xticklabels(["f=1", "f=2", "f=4"]); a.set_title(t, fontsize=9)
    a1.set_ylabel("ms (mean ± 95 % CI)")
    a1.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "f6_latency.png"); plt.close(fig)


def e4():
    d = load("e4_jitter")
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    js = ["none", "exp5", "exp20", "bursty"]
    kinds = [k for k in ("fixed", "chen", "chen-r", "phi") if any(x["det"] == k for x in d)]
    width = 0.8 / len(kinds)
    cols, labs = C, NAMES
    for j, kind in enumerate(kinds):
        mis, thr = [], []
        for jit in js:
            g = next(x for x in d if x["jitter"] == jit and x["det"] == kind)
            mis.append(A.stat([len(r["mistakes"]) for r in g["runs"]]))
            thr.append(A.stat([r["completed"] / r["makespan"] for r in g["runs"]]))
        pos = [i + (j - (len(kinds) - 1) / 2) * width for i in range(len(js))]
        a1.bar(pos, [m for m, _ in mis], width, color=cols[kind], label=labs[kind])
        a2.bar(pos, [m for m, _ in thr], width, yerr=[c for _, c in thr], capsize=2, color=cols[kind])
    for a in (a1, a2):
        a.set_xticks(range(len(js))); a.set_xticklabels(js); a.set_xlabel("jitter profile")
    a1.set_yscale("log")
    a1.set_ylabel("wrong suspicions per run"); a1.set_title("(a) Mistakes (one 0.3 s pause per run)", fontsize=9)
    a2.set_ylabel("tasks/s"); a2.set_title("(b) Job throughput", fontsize=9)
    a2.set_ylim(150, None)
    a1.set_ylim(0.7, 2000)
    a1.legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "f7_jitter.png"); plt.close(fig)


def e5():
    d = load("e5_policy")
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    pf = load("e5a_prefetch") or []
    ks = [g["k"] for g in pf]
    thr = [A.stat([A.x_pre(r) for r in g["runs"]]) for g in pf]
    rec = [A.stat(A.recovery(g["runs"]), 1000) for g in pf]
    a1.errorbar(ks, [m for m, _ in thr], yerr=[c for _, c in thr], marker="o", ms=4, capsize=3, color="#1f77b4")
    a1.set_xscale("log", base=2); a1.set_xlabel("prefetch depth k"); a1.set_ylabel("failure-free tasks/s", color="#1f77b4")
    a3 = a1.twinx(); a3.grid(False)
    a3.errorbar(ks, [m for m, _ in rec], yerr=[c for _, c in rec], marker="s", ms=4, capsize=3, color="#d62728",
                label="measured")
    a3.plot(ks, [g.get("model_R", float("nan")) for g in pf], "k--", lw=0.9, label="model")
    a3.set_ylabel("recovery latency (ms)", color="#d62728")
    a3.legend(fontsize=7, loc="upper left")
    a1.set_title("(a) Prefetch depth: throughput vs recovery", fontsize=9)

    labels = {("static", False): "static, no FT", ("static", True): "static + FT", ("pull", True): "pull + FT"}
    whens = [None, 2.0, 10.0]
    wl = ["no crash", "crash at 2 s", "crash at 10 s"]
    width = 0.25
    for j, key in enumerate(labels):
        vals = []
        for w in whens:
            g = next(x for x in d["policy"] if (x["policy"], x["ft"]) == key and x["crash_at"] == w)
            ms = [r["makespan"] for r in g["runs"]]
            vals.append(A.stat(ms) if all(m is not None for m in ms) else (None, None))
        pos = [i + (j - 1) * width for i in range(len(whens))]
        for p, (m, c) in zip(pos, vals):
            if m is None:
                a2.text(p, 0.5, "never\nfinishes", ha="center", fontsize=5.5, rotation=90, color="#b00000")
        a2.bar(pos, [m or 0 for m, _ in vals], width, yerr=[c or 0 for _, c in vals], capsize=2,
               label=labels[key], color=["#999", "#1f77b4", "#d62728"][j])
    a2.set_xticks(range(3)); a2.set_xticklabels(wl); a2.set_ylabel("makespan (s)")
    a2.set_title("(b) Policy and crash time (task CV 0.5)", fontsize=9)
    a2.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    a2.set_ylim(0, 27)
    fig.tight_layout(); fig.savefig(FIG / "f8_policy.png"); plt.close(fig)


def e6():
    d = load("e6_endcrash")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    base = [r["makespan"] - r["qempty_t"] for r in d[0]["runs"]]
    labels, tails, cis, dets = ["no crash"], [], [], []
    m, c = A.stat(base, 1000); tails.append(m); cis.append(c); dets.append(None)
    for g in d[1:]:
        m, c = A.stat([r["makespan"] - r["qempty_t"] for r in g["runs"]], 1000)
        tails.append(m); cis.append(c); labels.append(NAMES[g["det"]])
        dets.append(A.stat(A.detection(g["runs"]), 1000)[0])
    cols = ["#999"] + [C[g["det"]] for g in d[1:]]
    ax.bar(range(len(tails)), tails, yerr=cis, capsize=3, color=cols)
    ax.axhline(tails[0], color="k", ls=":", lw=0.9, label="no-crash tail")
    for i, dd in enumerate(dets):
        if dd is not None:
            ax.plot([i - 0.35, i + 0.35], [dd] * 2, "k-", lw=1.6,
                    label="detection latency D" if i == 1 else None)
    ax.set_xticks(range(len(tails))); ax.set_xticklabels(labels)
    ax.set_ylabel("queue empty → job end (ms)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "f9_endcrash.png"); plt.close(fig)


def e7():
    d = load("e7_tasksize")
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, mk in ((1, "o"), (4, "s")):
        gs = [g for g in d if g["k"] == k]
        xs = [g["s_ms"] for g in gs]
        rec = [A.stat(A.recovery(g["runs"]), 1000) for g in gs]
        a1.errorbar(xs, [m for m, _ in rec], yerr=[c for _, c in rec], marker=mk, ms=4, capsize=3,
                    label=f"k = {k} measured")
        a1.plot(xs, [g.get("model_R", float("nan")) for g in gs], "k--" if k == 4 else "k:", lw=0.9,
                label=f"k = {k} model")
    for a in (a1, a2):
        a.set_xscale("log"); a.set_xlabel("nominal task time s (ms)")
    a1.set_yscale("log"); a1.set_ylabel("recovery latency (ms)"); a1.legend(fontsize=6.5)
    a1.set_title("(a) Recovery latency", fontsize=9)
    sizes = sorted({g["s_ms"] for g in d})
    ratio = []
    for sz in sizes:
        x1 = [A.x_pre(r) for g in d if g["s_ms"] == sz and g["k"] == 1 for r in g["runs"]]
        x4 = [A.x_pre(r) for g in d if g["s_ms"] == sz and g["k"] == 4 for r in g["runs"]]
        ratio.append(A.stat([a / (sum(x4) / len(x4)) * 100 for a in x1]))
    a2.errorbar(sizes, [m for m, _ in ratio], yerr=[c for _, c in ratio], marker="o", ms=4, capsize=3,
                color="#1f77b4")
    a2.axhline(100, ls="--", color="gray", lw=0.8)
    a2.set_ylabel("throughput k = 1 (% of k = 4)")
    a2.set_title("(b) Cost of no prefetching", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "f10_tasksize.png"); plt.close(fig)


if __name__ == "__main__":
    e2_data = load("e2_qos")
    if e2_data:                       # add the model predictions to E3 and E5 before drawing them
        CACHE["e3_crash"], CACHE["e5a_prefetch"] = load("e3_crash"), load("e5a_prefetch")
        CACHE["e7_tasksize"] = load("e7_tasksize")
        A.annotate(CACHE["e3_crash"], CACHE["e5a_prefetch"], e2_data, e7=CACHE["e7_tasksize"])
    architecture()
    e1(); e2(); e3(); e4(); e5(); e6(); e7()
    print("figures written to", FIG)
