"""Reverse-order H2 recheck, extended correctness, then full14-case regression."""
import gc
import json
import os
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parents[1]
assert os.environ.get("NSA_VALIDATION_GUARD_PID")==str(os.getppid())
stages=[
    ["calibrate_oj_protocol.py","--modules","submissions/nsa140.py","submissions/nsa138.py","submissions/nsa128.py",
     "--ids","13","--seeds","577","--modes","public","current","--profiles","cupti","--rounds","4",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa138_140_recheck.jsonl"],
    ["stress_nsa_h2_small.py","--modules","submissions/nsa128.py","submissions/nsa140.py",
     "--output","results/nsa140_stress.jsonl"],
    ["calibrate_oj_protocol.py","--modules","submissions/nsa128.py","submissions/nsa140.py",
     "--ids",*map(str,range(1,15)),"--seeds","581","--modes","public","current","--profiles","cupti","--rounds","2",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa140_full.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa128.py","submissions/nsa140.py",
     "--ids",*map(str,range(1,15)),"--output-dir","results/sources140"],
]
for args in stages:
    opt="--output" if "--output" in args else "--output-dir"
    assert not (root/args[args.index(opt)+1]).exists(),"Refuse existing artifacts"
sys.path.insert(0,str(root/"tests"))
for args in stages:
    print(json.dumps(dict(stage="begin",script=args[0])),flush=True)
    sys.argv=[str(root/"tests"/args[0]),*args[1:]]
    try: runpy.run_path(sys.argv[0],run_name="__main__")
    except SystemExit as e:
        if e.code not in (None,0): raise
    gc.collect()
    print(json.dumps(dict(stage="complete",script=args[0])),flush=True)
