"""Source-bound two-batch outcome and checker-threshold verification."""
import hashlib,json,math,statistics
from pathlib import Path
from reconcile_oj_calibration import checker_points,score
root=Path(__file__).resolve().parents[1]
dest=root/'results/2026-09-09'
files={109:'probe_nsa109_s2_scalar_probability.py',119:'probe_nsa119_d128_disjoint_reduce.py',
120:'probe_nsa120_d128_disjoint_reduce_barrier_control.py',121:'probe_nsa121_d128_compact_reduce.py'}
hashes={f'submissions/nsa{v}.py':hashlib.sha256((root/'probes'/f).read_bytes()).hexdigest() for v,f in files.items()}
checks=samples=0
for name,versions,seed,reports in [
 ('nsa119_120.jsonl',(109,120,119),439,{119:'nsa119_comparison.json',120:'nsa120_comparison.json'}),
 ('nsa121.jsonl',(121,120,109),443,{121:'nsa121_comparison.json',120:'nsa120_recheck_comparison.json'})]:
    rows=[json.loads(x) for x in (dest/name).read_text().splitlines()]
    assert len(rows)==6
    assert {(r['module'],r['mode']) for r in rows}=={(f'submissions/nsa{v}.py',m) for v in versions for m in ('public','current')}
    for r in rows:
        assert r['source_sha256']==hashes[r['module']] and r['correct']
        assert r['case_id']==6 and r['seed']==seed
        assert len(r['samples_us']['cupti'])==len(r['guard_samples']['cupti'])==3
        for g in r['guard_samples']['cupti']:
            assert max(g['before_us'],g['after_us'])/min(g['before_us'],g['after_us'])-1<=.03
            assert .92<=g['reference_ratio']<=1.08
        if r['module'] in ('submissions/nsa119.py','submissions/nsa120.py'):
            v=r['module'].split('nsa')[1].split('.')[0]
            s=(dest/f'sources119_120/nsa{v}_case6.cu').read_text()
            s='\n'.join(x.rstrip() for x in s.splitlines()).rstrip()+'\n'
            assert hashlib.sha256(s.encode()).hexdigest()==r['kernel_source_sha256']
    for v,report in reports.items():
        ratios=[]
        for mode in ('public','current'):
            means={}
            for r in rows:
                if r['mode']==mode:
                    means[r['module']]=statistics.median(t/((g['before_us']+g['after_us'])/2)
                        for t,g in zip(r['samples_us']['cupti'],r['guard_samples']['cupti']))
            ratios.append(means[f'submissions/nsa{v}.py']/means['submissions/nsa109.py'])
        result=math.exp(sum(map(math.log,ratios))/2)
        assert math.isclose(result,json.loads((dest/report).read_text())['geometric_relative_time'],rel_tol=1e-12)
    checks+=6;samples+=18
oj=next(r for r in json.loads((dest/'oj109_final.json').read_text()) if r['meta']['id']==141658)
assert oj['meta']['status']=='Accepted' and oj['source']['sha256']==hashes['submissions/nsa109.py']
point=checker_points(oj)[6]
threshold=point['baseline_us']*(100/(point['score']+1)-1)
ratio=json.loads((dest/'nsa120_recheck_comparison.json').read_text())['geometric_relative_time']
assert score(point['baseline_us'],point['oj_us']*ratio)==84
print(json.dumps(dict(verification='PASS',reference_checks=checks,qualified_samples=samples,
    next_point_threshold_us=threshold,required_time_reduction=1-threshold/point['oj_us'],
    nsa120_projected_case_score=84,limitation='Projection is not a new OJ score; only case6 measured')))
