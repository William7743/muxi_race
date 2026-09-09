"""S4 explicit synchronization stress: fresh allocations/values, causal edges and fallbacks."""
import argparse
import hashlib
import json
import torch
from benchmark import ROOT, inputs, reference, load

p = argparse.ArgumentParser()
p.add_argument("--modules", nargs="+", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
modules = {name: load(ROOT/name) for name in a.modules}
shapes = ((2,512,1,64,16),(1,512,2,64,16),(1,513,1,64,16),
          (1,496,1,64,16),(1,64,1,64,16),(4,1024,1,64,16),
          (1,512,1,64,32),(1,512,1,128,16))
failures = 0
with (ROOT/a.output).open("x") as stream:
    for b,length,h,d,bs in shapes:
        case = dict(B=b,seq_len=length,kv_heads=h,q_heads=h*16,dim=d,
                    selected_blocks=4,block_size=bs,causal=1)
        data,args = inputs(case,751,"current")
        q,k,v,ix = data
        out = torch.empty_like(q)
        current = (torch.arange(length,device=q.device,dtype=torch.int32)//bs)[None,:,None]
        for update in range(3):
            torch.manual_seed(751+update)
            q.normal_().mul_(4 if update == 1 else 1)
            k.normal_().mul_(4 if update == 1 else 1)
            v.normal_().mul_(16 if update == 1 else 1)
            if update == 1:
                ix.fill_(length)
                ix[:,:,:,-1] = current
            elif update == 2:
                q.zero_()
                ix.fill_(length)
                ix[:,:,:,0] = current
                ix[:,:,:,-1] = torch.clamp(current-1,min=0)
                signs = (torch.arange(length,device=q.device)%2*2-1).to(v.dtype)
                v.copy_(signs[None,:,None,None].expand_as(v)*1024)
            assert ix.dtype == torch.int32 and all(t.is_contiguous() for t in data)
            assert ((ix == length)|((ix >= 0)&(ix < (length+bs-1)//bs))).all()
            expected = reference(*data,bs,True)
            assert torch.isfinite(expected).all()
            for name,mod in modules.items():
                out.fill_(float("nan"))
                mod.run_kernel(*data,out,*args)
                torch.cuda.synchronize()
                mismatch = int((~torch.isclose(out,expected,atol=.01,rtol=.01)).sum())
                failures += mismatch > 0
                row = dict(module=name,source_sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),
                    case=case,seed=751+update,update=update,index_dtype=str(ix.dtype),
                    nonfinite=int((~torch.isfinite(out)).sum()),mismatch_count=mismatch,
                    max_abs=float((out-expected).abs().max()),status="PASS" if mismatch==0 else "FAIL")
                stream.write(json.dumps(row)+"\n"); stream.flush()
                print(json.dumps(row),flush=True)
raise SystemExit(1 if failures else 0)
