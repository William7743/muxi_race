import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import XPUOJClient, load_credentials
import argparse

ns = argparse.Namespace(email=None, password=None)
email, pw = load_credentials(ns)
c = XPUOJClient(email, pw)

# usage: python test_probe.py solution_v59.py
fname = sys.argv[1] if len(sys.argv) > 1 else "solution_v59.py"
with open(fname, "r", encoding="utf-8") as f:
    code = f.read()

data = c.submit(code, contest_id=5, problem_order=1)
sid = data.get("submissionId") or data.get("id")
print("file=", fname, " submissionId=", sid)
sys.stdout.flush()
print("开始轮询...", flush=True)
for i in range(120):
    time.sleep(8)
    d = c.get_submission_detail(sid)
    if not d:
        continue
    meta = d.get("meta", d)
    status = meta.get("status")
    if status in ("Running", "Pending", "Judging"):
        print(f"[{i+1}] {status}", flush=True)
        continue
    results = meta.get("results") or {}
    ts = []
    for k, v in results.items():
        tm = v.get("time", 0)
        st = v.get("status", "?")
        ts.append(f"c{len(ts)+1}={tm/1000.0:.3f}({st})")
    print("cases:", " ".join(ts), flush=True)
    if status == "WrongAnswer":
        print("WA", flush=True); sys.exit(3)
    if status == "CompileError":
        print("CE", flush=True); sys.exit(4)
    if status == "Accepted":
        ds = meta.get("displayScore")
        print(f"ds={ds} | {' '.join(ts)}", flush=True)
        sys.exit(0)
    print("status=", status, flush=True)
    if status in ("TimeLimitExceeded", "MemoryLimitExceeded", "RuntimeError"):
        sys.exit(5)
