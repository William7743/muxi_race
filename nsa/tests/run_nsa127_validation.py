"""Run NSA127 stages after a handoff or supervised, observed peer completion.

--dry-run does not import Torch/TileLang or initialize the device.
The memory preflight is advisory; observed completion requires a live supervisor.
"""
import argparse
import gc
import json
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
stages = [
    ["stress_nsa127_bounds.py", "--modules", "submissions/nsa127.py", "submissions/nsa123.py",
     "--output", "results/nsa127_stress_handoff.jsonl"],
    ["calibrate_oj_protocol.py", "--modules", "submissions/nsa123.py", "submissions/nsa127.py",
     "--ids", *map(str, range(1,15)), "--seeds", "491", "--modes", "public", "current",
     "--profiles", "cupti", "--rounds", "2", "--guard-module", "submissions/nsa_v159.py",
     "--output", "results/nsa127_full_handoff.jsonl"],
    ["dump_sources.py", "--modules", "submissions/nsa123.py", "submissions/nsa127.py",
     "--ids", "4", "5", "7", "9", "14", "--output-dir", "results/sources127_handoff"],
]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dry-run", action="store_true")
p.add_argument("--handoff-confirmed", action="store_true",
               help="Use only after the peer/user confirms no further tests will start")
p.add_argument("--peer-terminal-observed", action="store_true",
               help="Prior workflows are terminal; requires the live exclusive supervisor")
a = p.parse_args()
if a.dry_run:
    print(json.dumps(dict(dry_run=True, stages=stages)))
    raise SystemExit(0)
if not a.handoff_confirmed and not a.peer_terminal_observed:
    p.error("--handoff-confirmed required: an individual batch exit or idle gap is insufficient")
if a.peer_terminal_observed and os.environ.get("NSA_VALIDATION_GUARD_PID") != str(os.getppid()):
    p.error("Observed completion requires the live exclusive supervisor")

# Refuse to overwrite either completed or interrupted artifacts.
for args in stages:
    option = "--output" if "--output" in args else "--output-dir"
    path = root / args[args.index(option)+1]
    if path.exists():
        raise RuntimeError(f"Preserve existing artifact instead of overwriting: {path}")
    if not (root/"tests"/args[0]).is_file():
        raise RuntimeError(f"Missing staged helper: {args[0]}")

cgroup = Path("/sys/fs/cgroup/memory")
limit = int((cgroup/"memory.limit_in_bytes").read_text())
used = int((cgroup/"memory.usage_in_bytes").read_text())
print(json.dumps(dict(stage="memory_preflight", limit_bytes=limit, used_bytes=used)), flush=True)
if limit-used < 24*(1024**3):
    raise RuntimeError("Need at least24GiB host-memory headroom; wait for peer handoff")

sys.path.insert(0, str(root/"tests"))
for args in stages:
    print(json.dumps(dict(stage="begin", script=args[0])), flush=True)
    sys.argv = [str(root/"tests"/args[0]), *args[1:]]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None,0):
            raise
    gc.collect()
    print(json.dumps(dict(stage="complete", script=args[0])), flush=True)
print(json.dumps(dict(workflow="complete", limitation="Local validation, not OJ score")), flush=True)
