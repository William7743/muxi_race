"""Current-input correctness checks; no timings and no OJ submission."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
from smoke_nsa import reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--include-large', action='store_true',
                   help='Also exercise the large-grid shared-probability specialization')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('edge_candidate', a.candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # G8 fallback, G32, both block sizes, S>1 sentinel padding.
    cases = [(1,128,1,16,128,1,16), (1,128,1,32,128,1,32),
             (1,128,2,16,128,1,16), (1,128,2,32,64,4,32)]
    if a.include_large:
        cases.append((2,1024,1,16,128,1,32))
    for b,l,h,hq,d,s,bs in cases:
        for mode in ('current','first','scaled','zero_query'):
            torch.manual_seed(335)
            q = torch.randn(b,l,hq,d,device='cuda',dtype=torch.float16)
            k = torch.randn(b,l,h,d,device='cuda',dtype=torch.float16)
            v = torch.randn_like(k)
            if mode == 'scaled':
                q.mul_(4)
                k.mul_(4)
            if mode == 'zero_query':
                q.zero_()
            idx = torch.full((b,l,h,s),l,dtype=torch.int32,device='cuda')
            blocks = torch.arange(l,device='cuda',dtype=torch.int32)//bs
            idx[:,:,:,0] = (torch.zeros_like(blocks) if mode=='first' else blocks)[None,:,None]
            out = torch.full_like(q, float('nan'))
            assert all(x.is_contiguous() for x in (q,k,v,idx,out))
            expected = reference(q,k,v,idx,bs,True).half()
            mod.run_kernel(q,k,v,idx,out,b,l,h,hq,d,s,bs,1)
            torch.cuda.synchronize()
            torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
            print(json.dumps(dict(case=[b,l,h,hq,d,s,bs],mode=mode,status='PASS',
                  max_abs=float((out.float()-expected.float()).abs().max()))),flush=True)


if __name__ == '__main__':
    main()
