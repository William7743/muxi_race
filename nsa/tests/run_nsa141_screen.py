"""Combined output first screen; refuse existing outputs and require supervisor."""
import gc
import json
import os
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parents[1]
assert os.environ.get("NSA_VALIDATION_GUARD_PID")==str(os.getppid())
stages=[
    ["calibrate_oj_protocol.py","--modules","submissions/nsa138.py","submissions/nsa141.py",
     "--ids","6","--seeds","593","--modes","public","current","--profiles","cupti","--rounds","3",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa141_screen.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa138.py","submissions/nsa141.py",
     "--ids","6","--output-dir","results/artifacts141","--host-source","--artifact-metadata"],
]
for args in stages:
    opt="--output" if "--output" in args else "--output-dir"
    assert not (root/args[args.index(opt)+1]).exists()
sys.path.insert(0,str(root/"tests"))
for args in stages:
    print(json.dumps(dict(stage="begin",script=args[0])),flush=True)
    sys.argv=[str(root/"tests"/args[0]),*args[1:]]
    try: runpy.run_path(sys.argv[0],run_name="__main__")
    except SystemExit as e:
        if e.code not in (None,0): raise
    gc.collect()
    print(json.dumps(dict(stage="complete",script=args[0])),flush=True)
