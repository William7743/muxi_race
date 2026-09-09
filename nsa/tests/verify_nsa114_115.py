"""CPU audit of source-bound reference and guarded timing evidence."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
dest=root/'results/2026-09-09'
hashes={}
for version,filename in [(109,'probe_nsa109_s2_scalar_probability.py'),
                         (114,'probe_nsa114_d128_direct_output.py'),
                         (115,'probe_nsa115_d128_row_reciprocal.py')]:
    hashes[f'submissions/nsa{version}.py']=hashlib.sha256((root/'probes'/filename).read_bytes()).hexdigest()
checks=0
for version,qualified in [(114,0),(115,2)]:
    rows=[json.loads(x) for x in (dest/f'nsa{version}.jsonl').read_text().splitlines()]
    assert len(rows)==4
    assert {(r['module'],r['mode']) for r in rows}=={
        (m,mode) for m in ('submissions/nsa109.py',f'submissions/nsa{version}.py')
        for mode in ('public','current')}
    good=0
    for r in rows:
        assert r['source_sha256']==hashes[r['module']] and r['correct']
        assert r['case_id']==6
        assert len(r['guard_samples']['cupti'])==3
        for g in r['guard_samples']['cupti']:
            good+=(max(g['before_us'],g['after_us'])/min(g['before_us'],g['after_us'])-1<=.03
                   and .92<=g['reference_ratio']<=1.08)
    assert good==qualified
    summary=json.loads((dest/f'nsa{version}_comparison.json').read_text())
    assert summary['qualified_samples']==good and summary['geometric_relative_time'] is None
    checks+=len(rows)
rows=[json.loads(x) for x in (dest/'nsa115_stress.jsonl').read_text().splitlines()]
assert len(rows)==24
expected={(m,l,h,u) for m in ('submissions/nsa109.py','submissions/nsa115.py')
          for l,h in ((1024,1),(1056,1),(1025,1),(1024,2)) for u in range(3)}
assert {(r['module'],r['case']['seq_len'],r['case']['kv_heads'],r['update']) for r in rows}==expected
for r in rows:
    assert r['source_sha256']==hashes[r['module']]
    assert r['status']=='PASS' and r['mismatch_count']==0 and r['nonfinite']==0
checks+=len(rows)
print(json.dumps(dict(verification='PASS', reference_checks=checks,
    nsa114_candidate_checks=2,nsa115_candidate_checks=14,
    usable_timing_pairs=0,oj_submitted=False,limitation='No full-suite certification or demonstrated performance improvement')))

