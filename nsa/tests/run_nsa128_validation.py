"""Stage NSA128 validation under run_exclusive_validation.py; dry-run is CPU-only."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
stages = [
    ["calibrate_oj_protocol.py","--modules","submissions/nsa127.py","submissions/nsa128.py","submissions/nsa129.py",
     "--ids","12","--seeds","503","--modes","public","current","--profiles","cupti","--rounds","3",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa128_s8.jsonl"],
    ["stress_nsa128_s8.py","--modules","submissions/nsa127.py","submissions/nsa128.py",
     "--output","results/nsa128_stress.jsonl"],
    ["calibrate_oj_protocol.py","--modules","submissions/nsa127.py","submissions/nsa128.py",
     "--ids",*map(str,range(1,15)),"--seeds","509","--modes","public","current","--profiles","cupti","--rounds","2",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa128_full.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa127.py","submissions/nsa128.py",
     "--ids",*map(str,range(1,15)),"--output-dir","results/sources128"],
]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--dry-run",action="store_true")
a = p.parse_args()
if a.dry_run:
    print(json.dumps(dict(dry_run=True,stages=stages)))
    raise SystemExit(0)
if os.environ.get("NSA_VALIDATION_GUARD_PID") != str(os.getppid()):
    p.error("Requires a live exclusive supervisor")
for args in stages:
    option = "--output" if "--output" in args else "--output-dir"
    target = root/args[args.index(option)+1]
    if target.exists():
        raise RuntimeError(f"Refuse existing artifact: {target}")
    if not (root/"tests"/args[0]).is_file():
        raise RuntimeError(f"Missing helper: {args[0]}")
sys.path.insert(0,str(root/"tests"))
for args in stages:
    print(json.dumps(dict(stage="begin",script=args[0])),flush=True)
    sys.argv = [str(root/"tests"/args[0]),*args[1:]]
    try:
        runpy.run_path(sys.argv[0],run_name="__main__")
    except SystemExit as e:
        if e.code not in (None,0):
            raise
    gc.collect()
    print(json.dumps(dict(stage="complete",script=args[0])),flush=True)
print(json.dumps(dict(workflow="complete",limitation="Local validation, not OJ score")),flush=True)
