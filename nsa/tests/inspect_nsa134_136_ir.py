"""Extract source-bound captured134/136 device payloads to test binary equality."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
for v in (134,136):
    m=json.loads((root/f"results/artifacts134_137/nsa{v}_case11.artifact.json").read_text())
    subprocess.run([sys.executable,str(root/"tests/inspect_library_ir.py"),
        "--source",f"results/artifacts134_137/nsa{v}_case11.cu",
        "--library",m["library_path"],"--expected-library-sha256",m["library_sha256"],
        "--expected-source-sha256",m["cuda_sha256"],
        "--output-prefix",f"results/ir134_136/nsa{v}_case11","--device-sections"],check=True)
print("134_136_EXTRACTION_COMPLETE")

