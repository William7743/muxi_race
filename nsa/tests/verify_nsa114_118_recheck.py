"""Audit the two completed low-interference batches and generated sources."""
import hashlib,json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/'results/2026-09-09'
files={109:'probe_nsa109_s2_scalar_probability.py',114:'probe_nsa114_d128_direct_output.py',
       115:'probe_nsa115_d128_row_reciprocal.py',116:'probe_nsa116_d128_v_swizzle16.py',
       117:'probe_nsa117_d128_v_swizzle4.py',118:'probe_nsa118_d128_block_barrier_control.py'}
hashes={f'submissions/nsa{v}.py':hashlib.sha256((root/'probes'/name).read_bytes()).hexdigest() for v,name in files.items()}
checks=samples=0
for filename,versions,seed,rounds in [('nsa114_115_recheck.jsonl',(109,115,114),431,5),
                                    ('nsa116_118.jsonl',(109,118,116,117),433,3)]:
    rows=[json.loads(x) for x in (dest/filename).read_text().splitlines()]
    assert len(rows)==len(versions)*2
    assert {(r['module'],r['mode']) for r in rows}=={(f'submissions/nsa{v}.py',m) for v in versions for m in ('public','current')}
    for r in rows:
        assert r['source_sha256']==hashes[r['module']] and r['correct']
        assert r['case_id']==6 and r['seed']==seed
        gs=r['guard_samples']['cupti']
        assert len(gs)==rounds
        for g in gs:
            assert max(g['before_us'],g['after_us'])/min(g['before_us'],g['after_us'])-1<=.03
            assert .92<=g['reference_ratio']<=1.08
        if filename=='nsa114_115_recheck.jsonl':
            v=r['module'].split('nsa')[1].split('.')[0]
            cuda=(dest/f'sources114_115/nsa{v}_case6.cu').read_text()
            cuda='\n'.join(line.rstrip() for line in cuda.splitlines()).rstrip()+'\n'
            assert hashlib.sha256(cuda.encode()).hexdigest()==r['kernel_source_sha256']
    for v in versions:
        if v==109: continue
        ratios=[]
        for mode in ('public','current'):
            rel={}
            for r in rows:
                if r['mode']!=mode:continue
                rel[r['module']]=statistics.median(t/((g['before_us']+g['after_us'])/2)
                    for t,g in zip(r['samples_us']['cupti'],r['guard_samples']['cupti']))
            ratios.append(rel[f'submissions/nsa{v}.py']/rel['submissions/nsa109.py'])
        value=math.exp(sum(map(math.log,ratios))/len(ratios))
        name=f'nsa{v}_recheck_comparison.json' if v in (114,115) else f'nsa{v}_comparison.json'
        stored=json.loads((dest/name).read_text())
        assert math.isclose(value,stored['geometric_relative_time'],rel_tol=1e-12)
    checks+=len(rows);samples+=len(rows)*rounds
device=json.loads((dest/'device115/comparison.json').read_text())
assert not device['same_device_object'] and not device['same_text']
for v in (109,115):
    info=device['versions'][str(v)]
    assert info['source_sha256']==hashes[f'submissions/nsa{v}.py']
    payload=(dest/f'device115/nsa{v}_case6.device.text').read_bytes()
    text=info['device_metadata']['sections']['.text']
    assert text['bytes']==len(payload)==5632
    assert text['sha256']==hashlib.sha256(payload).hexdigest()
print(json.dumps(dict(verification='PASS',reference_checks=checks,qualified_samples=samples,total_samples=samples,
                     limitation='Two local target-case batches; not a new OJ score or full-suite proof')))
