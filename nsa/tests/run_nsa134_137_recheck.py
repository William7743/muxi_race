"""S4 explicit synchronization first screen and source-bound library capture."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
stages=[
    ["calibrate_oj_protocol.py","--modules","submissions/nsa137.py","submissions/nsa136.py","submissions/nsa134.py","submissions/nsa128.py",
     "--ids","11","--seeds","547","--modes","public","current","--profiles","cupti","--rounds","4",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa134_137_screen.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa137.py","submissions/nsa136.py","submissions/nsa134.py","submissions/nsa128.py",
     "--ids","11","--output-dir","results/artifacts134_137","--host-source","--artifact-metadata"],
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
reduce_header=Path("/opt/tilelang-metax-v0.1.10/src/tl_templates/maca/reduce.h")
reduce_hash=hashlib.sha256(reduce_header.read_bytes()).hexdigest()
print(json.dumps(dict(reduce_header=str(reduce_header),sha256=reduce_hash)),flush=True)
for args in stages:
    print(json.dumps(dict(stage="begin",script=args[0])),flush=True)
    sys.argv=[str(root/"tests"/args[0]),*args[1:]]
    try:
        runpy.run_path(sys.argv[0],run_name="__main__")
    except SystemExit as e:
        if e.code not in (None,0): raise
    gc.collect()
    print(json.dumps(dict(stage="complete",script=args[0])),flush=True)
assert hashlib.sha256(reduce_header.read_bytes()).hexdigest()==reduce_hash
print(json.dumps(dict(workflow="complete",limitation="Target screen only, not full validation")),flush=True)
