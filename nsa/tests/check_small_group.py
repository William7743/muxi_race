"""Full-output G8 verification, no score inference or baseline execution."""
import importlib.util
import json
import sys
import torch
from smoke_nsa import reference


def main():
    spec = importlib.util.spec_from_file_location('small_group',sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for d in (32,64,128):
        for s in (1,4):
            b,l,h,hq,bs = 1,128,2,16,32
            args = b,l,h,hq,d,s,bs,1
            for seed in (10,11):
                torch.manual_seed(seed)
                q = torch.randn(b,l,hq,d,device='cuda',dtype=torch.float16)
                k = torch.randn(b,l,h,d,device='cuda',dtype=torch.float16)
                v = torch.randn_like(k)
                bi = torch.full((b,l,h,s),l,dtype=torch.int32)
                for t in range(l):
                    for ih in range(h):
                        choices = torch.randperm(t//bs+1)[:s]
                        bi[0,t,ih,:len(choices)] = choices
                bi = bi.cuda().contiguous()
                out = torch.full_like(q,float('nan'))
                expected = reference(q,k,v,bi,bs,True).half()
                print(json.dumps(dict(case=args,seed=seed,phase='run')),flush=True)
                mod.run_kernel(q,k,v,bi,out,*args)
                torch.cuda.synchronize()
                torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
                print(json.dumps(dict(case=args,seed=seed,status='PASS',
                    max_abs_error=(out.float()-expected.float()).abs().max().item())),flush=True)


if __name__ == '__main__':
    main()
