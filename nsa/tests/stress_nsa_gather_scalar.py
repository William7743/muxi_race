"""S2/S4 regression: H1/H2, partial blocks, sentinel slots and in-place updates."""
import argparse
import hashlib
import json
import torch
from benchmark import ROOT, inputs, reference, load

p = argparse.ArgumentParser()
p.add_argument('--modules', nargs='+', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
mods = {name: load(ROOT/name) for name in a.modules}
failed = 0
with (ROOT/a.output).open('w') as stream:
    for blocks in (2, 4):
        for batch, length, heads in ((1,64,1),(2,257,1),(1,512,2),(1,513,2)):
            case = dict(B=batch, seq_len=length, kv_heads=heads, q_heads=heads*16,
                        dim=64, selected_blocks=blocks, block_size=16, causal=1)
            data, args = inputs(case, 509, 'current')
            q,k,v,ix = data
            assert ix.dtype == torch.int32 and all(t.is_contiguous() for t in data)
            output = torch.empty_like(q)
            for update in range(3):
                torch.manual_seed(509+update)
                q.normal_().mul_(4 if update else 1)
                k.normal_().mul_(4 if update else 1)
                v.normal_().mul_(16 if update else 1)
                if update:
                    ix.fill_(length)
                    slot = 0 if update == 1 else -1
                    ix[:,:,:,slot] = (torch.arange(length,device=q.device,dtype=torch.int32)//16)[None,:,None]
                expected = reference(*data, 16, True)
                assert torch.isfinite(expected).all()
                for name,mod in mods.items():
                    output.fill_(float('nan'))
                    mod.run_kernel(*data,output,*args)
                    torch.cuda.synchronize()
                    mismatch = int((~torch.isclose(output,expected,atol=.01,rtol=.01)).sum())
                    row = dict(module=name, source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                               case=case, update=update, seed=509+update,
                               index_mode='current_random' if update == 0 else ('current_first_only' if update == 1 else 'current_last_only'),
                               mismatch_count=mismatch, status='PASS' if mismatch == 0 else 'FAIL',
                               nonfinite=int((~torch.isfinite(output)).sum()), max_abs=float((output-expected).abs().max()))
                    failed += mismatch != 0
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    print(json.dumps(row),flush=True)
raise SystemExit(1 if failed else 0)
