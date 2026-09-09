"""Resume missing132/133 extraction using captured runtime (not assumed cache) paths."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
for v in (132,133):
    m=json.loads((root/f"results/artifacts132_133/nsa{v}_case6.artifact.json").read_text())
    subprocess.run([sys.executable,str(root/"tests/inspect_library_ir.py"),
        "--source",f"results/artifacts132_133/nsa{v}_case6.cu",
        "--library",m["library_path"],
        "--expected-library-sha256",m["library_sha256"],
        "--expected-source-sha256",m["cuda_sha256"],
        "--output-prefix",f"results/ir132_133/nsa{v}_case6",
        "--device-sections"],check=True)
print("CAPTURED_RUNTIME_LIBRARIES_EXTRACTED")

