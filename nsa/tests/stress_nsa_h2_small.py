"""H2 scalar-output branch and dispatch boundaries, including current partial blocks.

Reference/test tensor arithmetic is PyTorch; the submitted kernel is TileLang.
These are independently generated contract-range inputs, not hidden OJ data.
"""
import argparse
import hashlib
import json
import torch
from benchmark import ROOT, inputs, reference, load

p = argparse.ArgumentParser()
p.add_argument('--modules', nargs='+', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
modules = {name: load(ROOT / name) for name in a.modules}
failed = 0
with (ROOT / a.output).open('w') as stream:
    for batch, length in ((1,64), (1,128), (1,256), (1,257), (1,512), (1,513), (2,256), (2,257)):
        case = dict(B=batch, seq_len=length, kv_heads=2, q_heads=32, dim=64,
                    selected_blocks=1, block_size=16, causal=1)
        data, args = inputs(case, 509, 'current')
        q, k, v, ix = data
        assert all(x.is_contiguous() for x in data) and ix.dtype == torch.int32
        output = torch.empty_like(q)
        for update in range(2):
            torch.manual_seed(509 + update)
            q.normal_().mul_(1 if update == 0 else 4)
            k.normal_().mul_(1 if update == 0 else 4)
            v.normal_().mul_(1 if update == 0 else 16)
            if update:
                # Same index storage, now all queries use their current block,
                # including the last partial block for non-aligned lengths.
                ix[:] = (torch.arange(length, device=q.device, dtype=torch.int32) // 16)[None,:,None,None]
            expected = reference(*data, 16, True)
            assert torch.isfinite(expected).all()
            for name, module in modules.items():
                output.fill_(float('nan'))
                module.run_kernel(*data, output, *args)
                torch.cuda.synchronize()
                mismatch = int((~torch.isclose(output, expected, atol=.01, rtol=.01)).sum())
                row = dict(module=name, source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                           case=case, update=update, seed=509+update,
                           scalar_gate_expected=batch*length*2 <= 1024,
                           status='PASS' if mismatch == 0 else 'FAIL', mismatch_count=mismatch,
                           nonfinite=int((~torch.isfinite(output)).sum()),
                           max_abs=float((output-expected).abs().max()))
                failed += mismatch != 0
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(row), flush=True)
raise SystemExit(1 if failed else 0)
