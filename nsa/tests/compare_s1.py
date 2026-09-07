"""Paired NSA local comparison; no OJ submission. Public D128/BS32 cases plus two GQA checks."""
import importlib.util
import json
import statistics
import time
import argparse
from datetime import datetime
from pathlib import Path
import torch
from smoke_nsa import reference

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'baselines/oj_115804.py')
    parser.add_argument('--candidate', type=Path, default=ROOT / 'probes/probe_nsa003_bs32_ck4.py')
    parser.add_argument('--suite', choices=['bs32','qfragment','qwide','public'], default='bs32')
    parser.add_argument('--seed', type=int, default=314)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'results' / datetime.now().strftime('local-%Y%m%d-%H%M%S-%f'))
    options = parser.parse_args()
    options.output_dir.mkdir(parents=True, exist_ok=False)
    mods = {'baseline': load('old', options.baseline),
            'candidate': load('new', options.candidate)}
    with open(ROOT / 'vendor/public_20260907/test_cases_nsa_fwd.json') as f:
        public_cases = json.load(f)
        cases = [x for x in public_cases if x['S'] == 1 and x['D'] == 128 and x['block_size'] == 32]
    cases += [dict(B=2,SEQ_LEN=512,H=h,HQ=hq,D=128,S=1,block_size=32,is_causal=True) for h,hq in ((2,16),(1,32))]
    if options.suite == 'qfragment':
        cases = [dict(B=4,SEQ_LEN=1024,H=1,HQ=16,D=d,S=1,block_size=bs,is_causal=True)
                 for d in (32,64) for bs in (16,32)]
    if options.suite == 'qwide':
        cases = [dict(B=b,SEQ_LEN=l,H=1,HQ=16,D=64,S=1,block_size=32,is_causal=True)
                 for b,l in ((1,128),(1,512),(1,1024),(2,512),(2,1024),(4,1024),(8,1024))]
        cases += [dict(B=2,SEQ_LEN=512,H=1,HQ=32,D=64,S=1,block_size=32,is_causal=True)]
    if options.suite == 'public':
        cases = public_cases
    for tc in cases:
        args = tuple(tc[x] for x in ('B','SEQ_LEN','H','HQ','D','S','block_size','is_causal'))
        b, length, h, hq, dim, selected, bs, causal = args
        torch.manual_seed(options.seed)
        q = torch.randn(b, length, hq, dim, device='cuda', dtype=torch.float16)
        k = torch.randn(b, length, h, dim, device='cuda', dtype=torch.float16)
        v = torch.randn_like(k)
        idx = torch.full((b, length, h, selected), length, dtype=torch.int32)
        for ib in range(b):
            for t in range(length):
                for ih in range(h):
                    choices = torch.randperm(t // bs + 1)[:selected]
                    idx[ib,t,ih,:choices.numel()] = choices
        idx = idx.sort(-1).values.cuda().contiguous()
        expected = reference(q,k,v,idx,bs,causal).half()
        outs = {name: torch.full_like(q,float('nan')) for name in mods}
        assert all(x.is_contiguous() for x in (q,k,v,idx,*outs.values()))
        times = {name: [] for name in mods}
        for name, mod in mods.items():
            print(json.dumps(dict(case=tc, candidate=name, phase='compile_correctness')),flush=True)
            mod.run_kernel(q,k,v,idx,outs[name],*args)
            torch.cuda.synchronize()
            torch.testing.assert_close(outs[name],expected,atol=1e-2,rtol=1e-2)
            mod.run_kernel(q,k,v,idx,outs[name],*args)
        torch.cuda.synchronize()
        for order in [('baseline','candidate'),('candidate','baseline'),('candidate','baseline'),('baseline','candidate')]:
            for name in order:
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    mods[name].run_kernel(q,k,v,idx,outs[name],*args)
                end.record()
                end.synchronize()
                times[name].append(start.elapsed_time(end)/10)
        medians = {name:statistics.median(xs) for name,xs in times.items()}
        result = dict(case=tc,seed=options.seed,status='PASS',call_event_ms=times,median_ms=medians,
                      baseline_over_candidate=medians['baseline']/medians['candidate'])
        print(json.dumps(result),flush=True)
        with open(options.output_dir / 'paired_s1_results.jsonl','a') as f:
            f.write(json.dumps(result)+'\n')


if __name__ == '__main__':
    main()
