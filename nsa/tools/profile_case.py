"""Profiler workload only: full correctness before a bounded capture region.

All profiler APIs and torch reference work stay outside submitted kernels.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from smoke_nsa import reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=['baseline','gather'], default='baseline')
    parser.add_argument('--source', type=Path, help='Explicit standalone candidate, overrides --candidate')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--selected', type=int, default=8)
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    source = args.source or (ROOT / ('baselines/nsa_20260817.py' if args.candidate == 'baseline'
                     else 'probes/probe_nsa002_gather.py'))
    source = source.resolve()
    spec = importlib.util.spec_from_file_location('profile_candidate', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b,l,h,hq,d,s,bs,causal = args.batch,args.length,1,16,args.dim,args.selected,16,1
    torch.manual_seed(2718)
    q = torch.randn(b,l,hq,d,device='cuda',dtype=torch.float16)
    k = torch.randn(b,l,h,d,device='cuda',dtype=torch.float16)
    v = torch.randn_like(k)
    bi = torch.full((b,l,h,s), l, dtype=torch.int32)
    for ib in range(b):
        for t in range(l):
            selected = torch.randperm(max(1,t//bs))[:s]
            bi[ib,t,0,:len(selected)] = selected
    bi = bi.sort(-1).values.cuda().contiguous()
    out = torch.empty_like(q)
    assert all(x.is_contiguous() for x in (q,k,v,bi,out))
    params = b,l,h,hq,d,s,bs,causal
    expected = reference(q,k,v,bi,bs,True).half()
    mod.run_kernel(q,k,v,bi,out,*params)
    torch.cuda.synchronize()
    torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
    for _ in range(3):
        mod.run_kernel(q,k,v,bi,out,*params)
    torch.cuda.synchronize()
    print(json.dumps(dict(candidate=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         params=params,correctness='PASS',seed=2718)),flush=True)
    if args.capture:
        torch.cuda.profiler.start()
    try:
        for _ in range(3):
            mod.run_kernel(q,k,v,bi,out,*params)
        torch.cuda.synchronize()
    finally:
        if args.capture:
            torch.cuda.profiler.stop()
    print('PROFILE_WORKLOAD_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
