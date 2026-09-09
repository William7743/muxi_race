"""D128 reciprocal fresh-input checks; run beside the calibrated benchmark helpers."""
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
failures = 0
with (ROOT/a.output).open('w') as stream:
    for batch,length,heads in ((2,1024,1),(2,1056,1),(2,1025,1),(2,1024,2)):
        c=dict(B=batch,seq_len=length,kv_heads=heads,q_heads=heads*16,dim=128,
               selected_blocks=1,block_size=32,causal=1)
        data,args=inputs(c,617,'current')
        q,k,v,ix=data
        out=torch.empty_like(q)
        for update in range(3):
            torch.manual_seed(617+update)
            q.normal_().mul_(4 if update==1 else 1)
            k.normal_().mul_(4 if update==1 else 1)
            v.normal_().mul_(16 if update==1 else 1)
            if update:
                ix[:,:,:,0]=(torch.arange(length,device=q.device,dtype=torch.int32)//32)[None,:,None]
            if update==2:
                q.zero_()
                signs=(torch.arange(length,device=q.device)%2*2-1).to(v.dtype)
                v.copy_(signs[None,:,None,None].expand_as(v)*1024)
            assert all(t.is_contiguous() for t in data)
            expected=reference(*data,32,True)
            assert torch.isfinite(expected).all()
            for name,mod in mods.items():
                out.fill_(float('nan'))
                mod.run_kernel(*data,out,*args)
                torch.cuda.synchronize()
                mismatch=int((~torch.isclose(out,expected,atol=.01,rtol=.01)).sum())
                failures+=mismatch>0
                row=dict(module=name,source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                         case=c,update=update,seed=617+update,mismatch_count=mismatch,
                         nonfinite=int((~torch.isfinite(out)).sum()),max_abs=float((out-expected).abs().max()),
                         status='PASS' if mismatch==0 else 'FAIL')
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(row),flush=True)
raise SystemExit(1 if failures else 0)

