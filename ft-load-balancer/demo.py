"""Short demonstration runs; saves each console output in logs/

  python3 demo.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
LOGS.mkdir(exist_ok=True)

DEMOS = [
    ("demo1_nofault.txt", "--tasks 1200"),
    ("demo2_freeze.txt", "--tasks 2400 --fault freeze:0:3 --fault freeze:1:3 --trace"),
    ("demo3_kill_restart.txt", "--tasks 2400 --fault kill:2:3:r1.5 --eof --trace"),
    ("demo4_pause_bursty.txt", "--tasks 2400 --jitter bursty --detector fixed --param 0.065 "
                               "--fault pause:3:3:0.3 --trace"),
]

for name, args in DEMOS:
    cmd = [sys.executable, str(HERE / "ftlb.py")] + args.split()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE).stdout
    text = f"$ python3 ftlb.py {args}\n{out}"
    (LOGS / name).write_text(text)
    print(text)
