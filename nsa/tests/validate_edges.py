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
    p.add_argument('--include-combined', action='store_true',
                   help='Exercise D32 vec16 and S2 gather dispatch with valid edge inputs')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('edge_candidate', a.candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # G8 fallback, G32, both block sizes, S>1 sentinel padding.
    cases = [(1,128,1,16,128,1,16), (1,128,1,32,128,1,32),
             (1,128,2,16,128,1,16), (1,128,2,32,64,4,32)]
    if a.include_large:
        cases.append((2,1024,1,16,128,1,32))
    if a.include_combined:
        cases.extend([(4,1024,1,16,32,1,16), (4,1024,1,16,32,1,32),
                      (2,512,1,16,64,2,16)])
    for b,l,h,hq,d,s,bs in cases:
        modes = ['current','first','scaled','zero_query']
        if a.include_combined and s > 1:
            modes.extend(['duplicate', 'two_valid'])
        for mode in modes:
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
            if mode == 'duplicate':
                idx[:,:,:,1] = blocks[None,:,None]
            elif mode == 'two_valid':
                idx[:,:,:,1] = 0
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
