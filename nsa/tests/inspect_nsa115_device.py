"""Bind cached device payload extraction to the actual library recorded by JIT."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
folder=root/'results/device115'
summaries={}
for version in (109,115):
    prefix=folder/f'nsa{version}_case6'
    bound=json.loads(prefix.with_suffix('.artifact.json').read_text())
    lib=Path(bound['library_path'])
    assert hashlib.sha256(lib.read_bytes()).hexdigest()==bound['library_sha256']
    subprocess.run([sys.executable,str(root/'tests/inspect_cached_ir.py'),
                    '--source',str(prefix.with_suffix('.cu')),
                    '--output-prefix',str(prefix),
                    '--cache-directory',str(lib.parent),
                    '--device-sections'],check=True)
    extracted=json.loads(prefix.with_suffix('.metadata.json').read_text())
    assert extracted['library_sha256']==bound['library_sha256']
    assert extracted['kernel_source_sha256']==bound['cuda_sha256']
    summaries[str(version)]=dict(source_sha256=bound['source_sha256'],
        kernel_source_sha256=extracted['kernel_source_sha256'],
        library_sha256=bound['library_sha256'],
        device_metadata=extracted['device_metadata'],bitcode_sha256=extracted['bitcode_sha256'],
        ir_sha256=extracted['ir_sha256'])
report=dict(versions=summaries,
    same_device_object=summaries['109']['device_metadata']['object_sha256']==summaries['115']['device_metadata']['object_sha256'],
    same_text=summaries['109']['device_metadata']['sections']['.text']==summaries['115']['device_metadata']['sections']['.text'],
    limitation='Exact cached object identity check; not actual hardware counters or a performance claim')
(folder/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
