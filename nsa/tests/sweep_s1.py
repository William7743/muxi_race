"""Exploratory S1 chunk sweep. In-memory source variants, NOT an OJ submission.

Only CK/CV compile-time constants change. Python/torch are test orchestration.
Candidate source is registered in linecache for TileLang source inspection.
"""
import argparse
import json
import linecache
import statistics
import types
from pathlib import Path
import torch
from smoke_nsa import reference

ROOT = Path(__file__).resolve().parents[1]


def variant(ck, cv):
    source = (ROOT / 'baselines/oj_115804.py').read_text()
    source = source.replace('CK = 2  #', f'CK = {ck}  #', 1)
    source = source.replace('CV = 4  #', f'CV = {cv}  #', 1)
    filename = str(ROOT / f'tests/generated_s1_ck{ck}_cv{cv}.py')
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    mod = types.ModuleType(f's1_ck{ck}_cv{cv}')
    mod.__file__ = filename
    exec(compile(source, filename, 'exec'), mod.__dict__)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--splits', default='2:4,1:2,2:2,1:4')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    splits = [tuple(map(int,pair.split(':'))) for pair in a.splits.split(',')]
    assert all(ck in (1,2,4,8) and cv in (1,2,4,8) for ck,cv in splits)
    variants = {f'ck{ck}_cv{cv}': variant(ck,cv) for ck,cv in splits}
    cases = [(4,1024,1,16,128,1,16,1),(2,2048,1,16,128,1,32,1)]
    with a.output.open('x') as log:
        for args in cases:
            b,l,h,hq,d,s,bs,causal = args
            torch.manual_seed(a.seed)
            q = torch.randn(b,l,hq,d,device='cuda',dtype=torch.float16)
            k = torch.randn(b,l,h,d,device='cuda',dtype=torch.float16)
            v = torch.randn_like(k)
            # Include current partial blocks: do not assume all blocks precede t//BS.
            counts = torch.arange(l,device='cuda') // bs + 1
            bi = (torch.rand(b,l,h,s,device='cuda') * counts[None,:,None,None]).int().contiguous()
            expected = reference(q,k,v,bi,bs,True).half()
            out = torch.empty_like(q)
            kernels = {}
            times = {}
            for name,mod in variants.items():
                print(json.dumps(dict(case=args,variant=name,phase='compile')),flush=True)
                kernel = mod.nsa_chunk128(b,hq,l,d,True,bs,hq//h,s)
                kernel(q,k,v,bi,out)
                torch.cuda.synchronize()
                torch.testing.assert_close(out,expected,atol=1e-2,rtol=1e-2)
                kernels[name] = kernel
                times[name] = []
            names = list(kernels)
            for order in [names,names[::-1],names[::-1],names]:
                for name in order:
                    kernels[name](q,k,v,bi,out)
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        kernels[name](q,k,v,bi,out)
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end)*1000/10)
            result = dict(case=args,seed=a.seed,status='PASS',direct_callable_us=times,
                          median_us={n:statistics.median(t) for n,t in times.items()})
            log.write(json.dumps(result)+'\n')
            log.flush()
            print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
