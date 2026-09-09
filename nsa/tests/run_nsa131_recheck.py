"""Independent reversed-order D128 bounds recheck and compiled-artifact capture."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
stages=[
    ["calibrate_oj_protocol.py","--modules","submissions/nsa131.py","submissions/nsa128.py",
     "--ids","3","--seeds","523","--modes","public","current","--profiles","cupti","--rounds","5",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa131_recheck.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa128.py","submissions/nsa130.py","submissions/nsa131.py",
     "--ids","3","6","--output-dir","results/artifacts130_131","--host-source","--artifact-metadata"],
]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--dry-run",action="store_true")
a=p.parse_args()
if a.dry_run:
    print(json.dumps(stages)); raise SystemExit(0)
if os.environ.get("NSA_VALIDATION_GUARD_PID")!=str(os.getppid()):
    p.error("Requires live exclusive supervisor")
for args in stages:
    option="--output" if "--output" in args else "--output-dir"
    if (root/args[args.index(option)+1]).exists():
        raise RuntimeError("Preserve existing artifacts; choose a fresh run")
sys.path.insert(0,str(root/"tests"))
for args in stages:
    print(json.dumps(dict(stage="begin",script=args[0])),flush=True)
    sys.argv=[str(root/"tests"/args[0]),*args[1:]]
    try:
        runpy.run_path(sys.argv[0],run_name="__main__")
    except SystemExit as e:
        if e.code not in (None,0): raise
    gc.collect()
    print(json.dumps(dict(stage="complete",script=args[0])),flush=True)
print(json.dumps(dict(workflow="complete",limitation="Target screen only, not full validation")),flush=True)


