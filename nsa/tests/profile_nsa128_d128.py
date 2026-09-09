"""Capture current D128 resource-pressure counters; not a timing benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
assert os.environ.get("NSA_VALIDATION_GUARD_PID")==str(os.getppid())
prefix=root/"results/profile128_d128_source"
dest=root/"results/profile128_d128"
assert not dest.exists() and not prefix.with_suffix(".metadata.json").exists()
metrics=["AP MMA Duty ratio","ISU stall cycles layout",
         "shared memory access efficiency","Global Memory Read bytes",
         "Global Memory Write bytes","Private Read Instructions","Private Write Instructions"]
command=["/opt/mcProfiler-ubuntu18.04/mcProfiler","perf_exec",
    "--cmdline","python tests/profile_controlled_kernel.py --module submissions/nsa128.py --case-id 6 --prefix results/profile128_d128_source --seed 0 --mode public --cold",
    "--cwd",str(root),"--kernelname","kernel","--casename","nsa128_d128_pressure",
    "--custom","--counts","1","--metrics",*metrics,"--output",str(dest)]
print(json.dumps(dict(command=command,limitation="Counters are not qualified latency or causal proof")),flush=True)
subprocess.run(command,check=True,cwd=root)
print("PROFILE_DRIVER_COMPLETE",flush=True)
