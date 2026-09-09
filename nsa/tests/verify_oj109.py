"""Reconcile NSA109's complete OJ feedback with its pre-submission local pairing."""
import hashlib
import json
from reconcile_oj_calibration import DEST, SOURCE_FILES, checker_points, qualified, score

parent = next(r for r in json.loads((DEST/'oj_calibration_details.json').read_text()) if r['meta']['id'] == 141648)
candidate = json.loads((DEST/'oj109_final.json').read_text())[0]
assert candidate['meta']['id'] == 141658 and candidate['meta']['status'] == 'Accepted'
assert candidate['missing_case_result_count'] == 0
pp, cp = checker_points(parent), checker_points(candidate)
rows = [json.loads(l) for l in (DEST/'nsa109_full.jsonl').read_text().splitlines() if l.strip()]
assert len(rows) == 28
modules = {'submissions/nsa103.py': (141648, parent), 'submissions/nsa109.py': (141658, candidate)}
for row in rows:
    sid, report = modules[row['module']]
    assert row['correct'] and row['seed'] == 211 and row['mode'] == 'public'
    assert row['source_sha256'] == report['source']['sha256']
    assert row['source_sha256'] == hashlib.sha256(SOURCE_FILES[sid].read_text(encoding='utf-8').encode()).hexdigest()
points = []
for cid in range(1,15):
    by_module = {r['module']: r for r in rows if r['case_id'] == cid}
    assert set(by_module) == set(modules)
    a, b = by_module['submissions/nsa103.py'], by_module['submissions/nsa109.py']
    assert {k:v for k,v in a['case'].items() if k != 'case_id'} == pp[cid]['config'] == cp[cid]['config']
    assert a['case'] == b['case'] and pp[cid]['baseline_us'] == cp[cid]['baseline_us']
    same = a['kernel_source_sha256'] == b['kernel_source_sha256']
    assert same == (cid != 10)
    qa, qb = qualified(a), qualified(b)
    assert qa['guard_ratio'] and qb['guard_ratio']
    ratio = qb['guard_ratio']/qa['guard_ratio']
    projected = pp[cid]['oj_us'] * (1 if same else ratio)
    points.append(dict(case_id=cid, same_local_cuda=same, parent_oj_us=pp[cid]['oj_us'],
                       candidate_oj_us=cp[cid]['oj_us'], delta_score=cp[cid]['score']-pp[cid]['score'],
                       local_guard_relative_time=ratio, projected_oj_us=projected,
                       projected_score=score(pp[cid]['baseline_us'],projected),
                       actual_score=cp[cid]['score']))
report = dict(submission_id=141658, actual_score=candidate['meta']['displayScore'],
              source_and_scoring_checks='PASS', local_reference_checks=len(rows),
              unchanged_paths=[p['case_id'] for p in points if p['same_local_cuda']],
              unchanged_path_score_delta=sum(p['delta_score'] for p in points if p['same_local_cuda'])/14,
              changed_path_score_delta=sum(p['delta_score'] for p in points if not p['same_local_cuda'])/14,
              fixed_parent_projection=sum(p['projected_score'] for p in points)/14,
              points=points,
              limitation='Observed best86.00, not88. Local source equality does not prove OJ binary identity. Printed timing resolution can obscure small changes; its internal rounding rule is unknown.')
(DEST/'oj109_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'points'}))
