"""Read captured libraries only, after timing workflow is terminal."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
guard=json.loads((root/"results/nsa141_guarded.json").read_text())
assert guard["exit_code"]==0 and guard["reason"] is None
for v in (138,141):
    prefix=f"results/artifacts141/nsa{v}_case6"
    artifact=json.loads((root/(prefix+".artifact.json")).read_text())
    subprocess.run([sys.executable,str(root/"tests/inspect_library_ir.py"),
        "--source",prefix+".cu","--output-prefix",f"results/ir141/nsa{v}_case6",
        "--library",artifact["library_path"],"--expected-library-sha256",artifact["library_sha256"],
        "--expected-source-sha256",artifact["cuda_sha256"],"--device-sections"],check=True,cwd=root)
