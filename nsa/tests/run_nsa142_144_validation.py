"""S8 address screens plus full combination and H2/S4 boundary regression."""
import gc
import json
import os
from pathlib import Path
import runpy
import sys
root=Path(__file__).resolve().parents[1]
assert os.environ.get("NSA_VALIDATION_GUARD_PID")==str(os.getppid())
stages=[
    ["calibrate_oj_protocol.py","--modules","submissions/nsa142.py","submissions/nsa143.py","submissions/nsa144.py",
     "--ids","12","--seeds","599","--modes","public","current","--profiles","cupti","--rounds","3",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa143_144_screen.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa142.py","submissions/nsa143.py","submissions/nsa144.py",
     "--ids","12","--output-dir","results/artifacts143_144","--host-source","--artifact-metadata"],
    ["stress_nsa_h2_small.py","--modules","submissions/nsa128.py","submissions/nsa142.py","--output","results/nsa142_h2_stress.jsonl"],
    ["stress_nsa134_s4_sync.py","--modules","submissions/nsa128.py","submissions/nsa142.py","--output","results/nsa142_s4_stress.jsonl"],
    ["calibrate_oj_protocol.py","--modules","submissions/nsa128.py","submissions/nsa142.py",
     "--ids",*map(str,range(1,15)),"--seeds","601","--modes","public","current","--profiles","cupti","--rounds","2",
     "--guard-module","submissions/nsa_v159.py","--output","results/nsa142_full.jsonl"],
    ["dump_sources.py","--modules","submissions/nsa128.py","submissions/nsa142.py",
     "--ids",*map(str,range(1,15)),"--output-dir","results/sources142"],
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
