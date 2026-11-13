# Fault-tolerant dynamic load balancer

Source code and raw results for the term paper *Detection, Redistribution and Recovery: Quantifying
Throughput Degradation, Recovery Latency and Overhead in a Fault-Tolerant Dynamic Load Balancer*
(Robert Bota, MSc Computer Science, University of Zimbabwe, MCS Parallel and Distributed Algorithms,
Practical Unit 5 Task Set 2, November 2026).

A coordinator and N worker processes talk over TCP. Workers send heartbeats every h seconds; the
coordinator runs one failure detector per worker (fixed timeout, Chen et al. 2002, Chen with a reset
after each mistake, or the φ accrual detector of Hayashibara et al. 2004), puts a suspected worker's
tasks back on the queue, and commits every task exactly once. A fault injector freezes, kills, pauses or
restarts workers, and a jitter emulator adds network delay.

## Files

| File | Purpose |
|---|---|
| `common.py` | message framing, jitter emulation, the CPU-bound task, statistics |
| `detectors.py` | failure detectors: fixed, Chen, Chen-reset, φ accrual |
| `ftlb.py` | coordinator, worker processes and fault injector (run `python3 ftlb.py --help`) |
| `qos.py` | offline detector QoS evaluation on recorded heartbeat traces (T_D, λ_M, P_A) |
| `analysis.py` | metrics and the analytical model of detection and recovery latency |
| `run_experiments.py` | runs experiments E1–E7 and writes `results/*.json` |
| `figures.py` | draws every figure in `figures/` from `results/` |
| `demo.py` | four short demonstration runs; console output saved in `logs/` |
| `results/` | raw results of the runs reported in the paper (319 runs, 5 interleaved repetitions) |
| `figures/` | the figures as they appear in the paper |

## Requirements

- Linux (CPU pinning uses `sched_setaffinity`; faults use `SIGSTOP`/`SIGKILL`)
- Python 3.10 or later (tested with 3.14); the experiments use only the standard library
- `matplotlib` for `figures.py` only: `pip install -r requirements.txt`

By default the coordinator is pinned to CPU 2 and the eight workers to CPUs 4–11. On a machine with
fewer cores pass `--cpus` and `--coord-cpu` to `ftlb.py` (for example `--cpus 1-4 --workers 4`), or
`--cpus none --coord-cpu none` to switch pinning off.

## Running

```bash
python3 ftlb.py --workers 8 --tasks 2400 --fault freeze:0:4.0 --trace   # one run, one crash at t = 4 s
python3 demo.py                                                          # four short demo runs
python3 run_experiments.py                                               # all experiments, about 110 min
python3 run_experiments.py --only E3 E4 --reps 2                         # a quick subset
python3 figures.py                                                       # redraw the figures
```

`run_experiments.py` overwrites `results/`. Copy the folder first if you want to keep the published
results for comparison.

## Experiments

| Id | What it measures | Result file |
|---|---|---|
| E1 | heartbeat overhead, h from 5 ms to 250 ms | `e1_overhead.json` |
| E2 | detector QoS on recorded traces under four jitter profiles | `e2_qos.json` |
| E3 | crash recovery, f = 0, 1, 2, 4 of 8 workers, three detectors | `e3_crash.json` |
| E4 | jitter profiles and a 0.3 s pause, four detectors | `e4_jitter.json` |
| E5a | prefetch depth k = 1–16 with one crash | `e5a_prefetch.json` |
| E5 | static vs pull assignment, with and without fault tolerance | `e5_policy.json` |
| E6 | crash at the moment the queue runs empty | `e6_endcrash.json` |
| E7 | task size s = 2, 20, 80 ms × k = 1, 4 | `e7_tasksize.json` |

The published runs were made on a shared, power-limited laptop, so absolute timings on another machine
will differ. The paper's threats-to-validity section discusses this.
