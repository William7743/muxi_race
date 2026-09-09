"""Bind pair experiments to exact sources and check archived coverage; do not rank."""
import hashlib
import json
from pathlib import Path
from reconcile_oj_calibration import qualified

root = Path(__file__).resolve().parents[1]
dest = root/'results/2026-09-09'
paths = {'submissions/nsa109.py':'probe_nsa109_s2_scalar_probability.py',
         'submissions/nsa110.py':'probe_nsa110_pair_scores_shared_reuse.py',
         'submissions/nsa111.py':'probe_nsa111_pair_first_v_early.py'}
hashes = {m:hashlib.sha256((root/'probes'/p).read_text(encoding='utf-8').encode()).hexdigest() for m,p in paths.items()}
def rows(name):
    return [json.loads(l) for l in (dest/name).read_text().splitlines() if l.strip()]
initial, confirm = rows('nsa110.jsonl'), rows('nsa110_111_confirm.jsonl')
stress = rows('nsa110_stress.jsonl') + rows('nsa111_stress.jsonl')
assert len(initial) == 4 and len(confirm) == 6 and len(stress) == 48
for row in initial+confirm+stress:
    assert row['source_sha256'] == hashes[row['module']]
for row in initial+confirm:
    assert row['correct'] and row['case_id'] == 12 and row['mode'] in ('public','current')
assert {(r['module'],r['seed'],r['mode']) for r in initial} == {
    (m,0,mode) for m in ('submissions/nsa109.py','submissions/nsa110.py') for mode in ('public','current')}
assert {(r['module'],r['seed'],r['mode']) for r in confirm} == {
    (m,137,mode) for m in paths for mode in ('public','current')}
expected = {(m,length,heads,u) for m in paths for length,heads in ((1024,1),(1040,1),(1025,1),(1024,2)) for u in range(4)}
assert {(r['module'],r['case']['seq_len'],r['case']['kv_heads'],r['update']) for r in stress} == expected
for r in stress:
    assert r['status'] == 'PASS' and r['mismatch_count'] == r['nonfinite'] == 0
    assert r['case']['B'] == 1 and r['case']['dim'] == 64 and r['case']['block_size'] == 16
    assert r['case']['selected_blocks'] == 8 and r['case']['causal'] == 1
resources = rows('nsa110_resources.jsonl')
assert len(resources) == 2 and all(r['attributes_status'] == r['occupancy_status'] == 0 for r in resources)
assert {r['module'] for r in resources} == set(paths)-{'submissions/nsa111.py'}
report = dict(status='PASS', reference_assertions=58,
              assertions_per_module={m:sum(r['module']==m for r in initial+confirm+stress) for m in paths},
              initial_qualified=sum(qualified(r)['accepted'] for r in initial), initial_samples=12,
              confirm_qualified=sum(qualified(r)['accepted'] for r in confirm), confirm_samples=30,
              hashes=hashes,
              limitations=['Counts are assertions, not distinct shapes. Performance qualification is separate.',
                           'Stress includes two target shapes and two fallback shapes, not all possible inputs.',
                           'Resource upper bounds are not achieved occupancy. Neither probe is OJ certified.'])
(dest/'pair110_111_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
