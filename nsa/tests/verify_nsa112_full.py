"""Audit complete NSA112/109 comparison against fixed actual OJ109 results."""
import hashlib
import json
from pathlib import Path
from reconcile_oj_calibration import checker_points, qualified, score

root = Path(__file__).resolve().parents[1]
dest = root/'results/2026-09-09'
rows = [json.loads(s) for s in (dest/'nsa112_full.jsonl').read_text().splitlines() if s.strip()]
paths = {'submissions/nsa109.py': 'probe_nsa109_s2_scalar_probability.py',
         'submissions/nsa112.py': 'probe_nsa112_pair_combined_reductions.py'}
hashes = {m: hashlib.sha256((root/'probes'/p).read_text(encoding='utf-8').encode()).hexdigest()
          for m,p in paths.items()}
assert len(rows) == 28
assert {(r['module'],r['case_id']) for r in rows} == {(m,c) for m in paths for c in range(1,15)}
oj = json.loads((dest/'oj109_final.json').read_text())[0]
assert oj['meta']['id'] == 141658 and oj['meta']['status'] == 'Accepted'
assert oj['source']['sha256'] == hashes['submissions/nsa109.py']
observed = checker_points(oj)
guard_hashes = {r['guard_source_sha256'] for r in rows}
old = [json.loads(s) for s in (dest/'nsa109_full.jsonl').read_text().splitlines() if s.strip()]
assert len(guard_hashes) == 1 and guard_hashes == {r['guard_source_sha256'] for r in old}
for r in rows:
    assert r['correct'] and r['source_sha256'] == hashes[r['module']]
    assert r['seed'] == 317 and r['mode'] == 'public'
    assert r['index_dtype'] == 'torch.int32'
    assert len(r['samples_us']['cupti']) == 3
    assert {k:v for k,v in r['case'].items() if k != 'case_id'} == observed[r['case_id']]['config']
points = []
for cid in range(1,15):
    a = next(r for r in rows if r['case_id']==cid and r['module']=='submissions/nsa109.py')
    b = next(r for r in rows if r['case_id']==cid and r['module']=='submissions/nsa112.py')
    same = a['kernel_source_sha256'] == b['kernel_source_sha256']
    assert same == (cid != 12), 'Unexpected code change outside target or missing target change'
    qa,qb = qualified(a),qualified(b)
    usable = qa['guard_ratio'] is not None and qb['guard_ratio'] is not None
    ratio = qb['guard_ratio']/qa['guard_ratio'] if usable else None
    # Never infer a score gain from timing noise on unchanged generated paths.
    projected_us = observed[cid]['oj_us'] * (1 if same else ratio) if same or usable else None
    points.append(dict(case_id=cid,same_local_cuda=same,parent=qa,candidate=qb,
                       qualified=usable,guard_relative_time=ratio,
                       parent_oj_us=observed[cid]['oj_us'],parent_score=observed[cid]['score'],
                       projected_us=projected_us,projected_score=score(observed[cid]['baseline_us'],projected_us)
                       if projected_us is not None else None))
report = dict(status='PASS',source_hashes=hashes,reference_checks=28,
              qualified_samples=sum(qualified(r)['accepted'] for r in rows),total_samples=84,
              all_points_qualified=all(p['qualified'] for p in points),
              changed_cuda_cases=[p['case_id'] for p in points if not p['same_local_cuda']],
              fixed_parent_projection=sum(p['projected_score'] for p in points)/14
              if all(p['projected_score'] is not None for p in points) else None,
              actual_parent_score=oj['meta']['displayScore'],points=points,
              single_case12_ideal_score_upper_bound=(sum(p['score'] for p in observed.values())-observed[12]['score']+100)/14,
              limitations=['Static identity and reference PASS do not certify OJ acceptance.',
                           'Projection assumes local relative speed transfers to OJ; calibrated errors can be large.',
                           'Ideal upper bound uses zero time/100 points on case12, not an achievable score.',
                           'Full14 shapes with one seed do not exhaust valid-input correctness.'])
(dest/'nsa112_full_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='points'}))
print(json.dumps(points[11]))
