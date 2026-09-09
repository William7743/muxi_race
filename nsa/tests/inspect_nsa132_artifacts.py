"""Read-only extraction of actual NSA128/132/133 libraries captured by recheck.

Uses staged inspect_cached_ir.py; writes only new analysis artifacts, never cache.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
helper = root / "tests/inspect_cached_ir.py"
assert hashlib.sha256(helper.read_bytes()).hexdigest() == "234685bc0e261d32b71e76f2a92576e8f735dc2c94fb41f5ccc85448b25bbfe0"
for version, case in ((128,6),(132,6),(133,6)):
    prefix = root / f"results/ir132_133/nsa{version}_case{case}"
    if any(prefix.parent.glob(prefix.name + ".*")):
        raise RuntimeError("Preserve previous inspection")
    metadata = json.loads((root / f"results/artifacts132_133/nsa{version}_case{case}.artifact.json").read_text())
    library = Path(metadata["library_path"])
    assert hashlib.sha256(library.read_bytes()).hexdigest() == metadata["library_sha256"]
    subprocess.run([sys.executable,str(helper),
        "--source",f"results/artifacts132_133/nsa{version}_case{case}.cu",
        "--cache-directory",str(library.parent),
        "--output-prefix",str(prefix),"--device-sections"],check=True)
    result = json.loads(prefix.with_suffix(".metadata.json").read_text())
    assert result["library_sha256"] == metadata["library_sha256"]
    assert result["kernel_source_sha256"] == metadata["cuda_sha256"]
print("SOURCE_BOUND_EXTRACTION_COMPLETE")



