"""Correctness-check and capture one kernel, with optional cache flush outside the window."""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import torch
from benchmark import ROOT, inputs, load, reference

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--module', required=True)
p.add_argument('--case-id', type=int, required=True)
p.add_argument('--prefix', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--mode', choices=['public', 'current'], default='public')
p.add_argument('--cold', action='store_true')
a = p.parse_args()
module = load(ROOT / a.module)
case = next(c for c in json.loads((ROOT / 'tests/oj_cases.json').read_text()) if c['case_id'] == a.case_id)
data, scalars = inputs(case, a.seed, a.mode)
out = torch.full_like(data[0], float('nan'))
ref = reference(*data, scalars[6], scalars[7])
fn = lambda: module.run_kernel(*data, out, *scalars)
fn()
torch.cuda.synchronize()
torch.testing.assert_close(out, ref, atol=.01, rtol=.01)
error = float((out - ref).abs().max())
prefix = ROOT / a.prefix
prefix.parent.mkdir(parents=True, exist_ok=True)
source = module._get_kernel(*scalars).get_kernel_source()
source = '\n'.join(line.rstrip() for line in source.splitlines()).rstrip() + '\n'
prefix.with_suffix('.cu').write_text(source)
metadata = dict(module=a.module, case=case, seed=a.seed, mode=a.mode, cold=a.cold,
                cache_flush_bytes=256000000 if a.cold else 0, warmup=10, capture_calls=1,
                correct=True, max_abs=error, torch_version=torch.__version__,
                source_sha256=hashlib.sha256((ROOT / a.module).read_bytes()).hexdigest(),
                kernel_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                expected_output_bytes=out.numel() * out.element_size(),
                timestamp_utc=datetime.now(timezone.utc).isoformat())
prefix.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
cache = torch.empty(64000000, device='cuda', dtype=torch.int32) if a.cold else None
for _ in range(10):
    fn()
torch.cuda.synchronize()
if cache is not None:
    cache.zero_()
    torch.cuda.synchronize()
torch.cuda.profiler.start()
fn()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
