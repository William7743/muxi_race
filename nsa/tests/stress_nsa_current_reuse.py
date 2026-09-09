"""Contract-shaped causal current-block, sentinel and same-storage input updates.

Independent regression data, not reconstructed hidden OJ inputs. Run next to
the pinned external benchmark.py. PyTorch math exists only in this test.
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
    for b, length, dim, blocks, bs in ((2,1024,128,1,32), (1,1024,64,8,16), (2,257,64,1,16)):
        case = dict(B=b, seq_len=length, kv_heads=1, q_heads=16, dim=dim,
                    selected_blocks=blocks, block_size=bs, causal=1)
        data, args = inputs(case, 137, 'current')
        q, k, v, ix = data
        assert ix.dtype == torch.int32 and all(x.is_contiguous() for x in data)
        out = torch.empty_like(q)
        for last_slot in (False, True):
            ix.fill_(length)
            ix[:, :, 0, -1 if last_slot else 0] = torch.arange(length, device=q.device, dtype=torch.int32) // bs
            for update in range(2):
                # Same storage, changed values; the output is never reused as a result.
                torch.manual_seed(137 + update)
                q.normal_().mul_(1 if update == 0 else 4)
                k.normal_().mul_(1 if update == 0 else 4)
                v.normal_().mul_(1 if update == 0 else 16)
                ref = reference(*data, bs, True)
                assert torch.isfinite(ref).all()
                for name, module in modules.items():
                    out.fill_(float('nan'))
                    module.run_kernel(*data, out, *args)
                    torch.cuda.synchronize()
                    mismatch = int((~torch.isclose(out, ref, atol=.01, rtol=.01)).sum())
                    row = dict(module=name, source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                               case=case, last_slot=last_slot, update=update, seed=137+update,
                               status='PASS' if mismatch == 0 else 'FAIL', mismatch_count=mismatch,
                               nonfinite=int((~torch.isfinite(out)).sum()), max_abs=float((out-ref).abs().max()))
                    failed += mismatch != 0
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    print(json.dumps(row), flush=True)
raise SystemExit(1 if failed else 0)
