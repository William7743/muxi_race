"""Extract embedded LLVM IR from a source-bound captured runtime library.

Adapted from staged inspect_cached_ir.py (William7743/NSA); supports first-JIT
/tmp libraries as well as persistent caches. Caller must supply captured hashes.

Uses only file reads and binary extraction tools; does not compile or run kernels.
The embedded IR is not the final machine ISA. Private access counters and runtime
attributes remain separate evidence of actual execution/storage.
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', required=True)
p.add_argument('--output-prefix', required=True)
p.add_argument('--library', required=True)
p.add_argument('--expected-library-sha256', required=True)
p.add_argument('--expected-source-sha256', required=True)
p.add_argument('--device-sections', action='store_true',
               help='Also hash embedded device code, constants and note sections')
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / a.source
prefix = root / a.output_prefix
prefix.parent.mkdir(parents=True, exist_ok=True)


def normalized_hash(path):
    text = '\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip() + '\n'
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


source_hash = normalized_hash(source)
assert source_hash == a.expected_source_sha256, 'Saved CUDA source differs from capture'
library = Path(a.library)
library_hash = file_hash(library)
assert library_hash == a.expected_library_sha256, 'Captured runtime library changed'
if any(prefix.parent.glob(prefix.name + '.*')):
    raise RuntimeError('Refuse to overwrite analysis artifacts')
bin_dir = Path(os.environ.get('MACA_PATH', '/opt/maca')) / 'mxgpu_llvm/bin'
fatbin, bitcode, ir = (prefix.with_suffix(suffix) for suffix in ('.fatbin', '.bc', '.ll'))
commands = []


def run(command):
    commands.append([str(x) for x in command])
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


run([bin_dir / 'llvm-objcopy', '--dump-section', f'.mc_fatbin={fatbin}',
     library, prefix.with_suffix('.copy.so')])
bundles = run([bin_dir / 'clang-offload-bundler', '--type=o', f'--input={fatbin}', '--list']).splitlines()
targets = [name for name in bundles if name.startswith('maca-') and name.endswith('-bc')]
if len(targets) != 1:
    raise RuntimeError(f'Expected one MACA bitcode bundle, found {targets}')
run([bin_dir / 'clang-offload-bundler', '--type=o', f'--input={fatbin}',
     f'--targets={targets[0]}', f'--output={bitcode}', '--unbundle'])
run([bin_dir / 'llvm-dis', bitcode, '-o', ir])
lines = ir.read_text().splitlines()
private_lines = [dict(line=i, text=line.strip()) for i, line in enumerate(lines, 1)
                 if 'addrspace(5)' in line or ('call ' in line and '@llvm.memcpy.p' in line
                                              and ('.p5.' in line or '.p5(' in line))]
device_metadata = None
if a.device_sections:
    device_target = targets[0][:-3]
    if device_target not in bundles:
        raise RuntimeError('Matching device object bundle is absent')
    device = prefix.with_suffix('.device')
    run([bin_dir / 'clang-offload-bundler', '--type=o', f'--input={fatbin}',
         f'--targets={device_target}', f'--output={device}', '--unbundle'])
    section_files = {name: prefix.with_suffix(f'.device{name}')
                     for name in ('.text', '.rodata', '.note')}
    command = [bin_dir / 'llvm-objcopy']
    for name, path in section_files.items():
        command += ['--dump-section', f'{name}={path}']
    run(command + [device, prefix.with_suffix('.device.copy')])
    device_metadata = dict(object_sha256=file_hash(device),
                           sections={name: dict(bytes=path.stat().st_size, sha256=file_hash(path))
                                     for name, path in section_files.items()})
assert file_hash(library) == library_hash, 'Cached library changed during inspection'
metadata = dict(source=a.source, kernel_source_sha256=source_hash,
                library_path=str(library), library_sha256=library_hash,
                fatbin_sha256=file_hash(fatbin), bitcode_sha256=file_hash(bitcode),
                ir_sha256=file_hash(ir), bundle_ids=bundles, commands=commands,
                private_ir_lines=private_lines,
                device_metadata=device_metadata,
                limitation='Embedded IR is not final ISA; counts are static source matches, not runtime accesses.')
prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(dict(source=a.source, kernel_source_sha256=source_hash,
                      private_ir_lines=private_lines)), flush=True)


