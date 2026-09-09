"""Verify archived coverage and source identities for the final selective scalar file."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dest = root/'results/2026-09-09'
def read(name):
    return [json.loads(x) for x in (dest/name).read_text().splitlines() if x.strip()]
full, stress = read('nsa108_full.jsonl'), read('nsa108_stress.jsonl')
assert len(full) == 28 and len(stress) == 48
paths = {'submissions/nsa103.py':'probe_nsa103_h2_small_scalar_output.py',
         'submissions/nsa108.py':'probe_nsa108_gather_selective_scalar.py'}
hashes = {name:hashlib.sha256((root/'probes'/file).read_text(encoding='utf-8').encode()).hexdigest() for name,file in paths.items()}
for row in full+stress:
    assert row['source_sha256'] == hashes[row['module']]
unchanged = []
for cid in range(1,15):
    group = {r['module']:r for r in full if r['case_id'] == cid}
    assert set(group) == set(paths)
    assert all(r['correct'] and r['seed'] == 42 and r['mode'] == 'public' for r in group.values())
    distinct = len({r['kernel_source_sha256'] for r in group.values()})
    if cid not in (10,11):
        assert distinct == 1
        unchanged.append(cid)
    else:
        assert distinct == 2
expected = {(s,b,l,h,u,m) for s in (2,4) for b,l,h in ((1,64,1),(2,257,1),(1,512,2),(1,513,2)) for u in range(3) for m in paths}
actual = {(r['case']['selected_blocks'],r['case']['B'],r['case']['seq_len'],r['case']['kv_heads'],r['update'],r['module']) for r in stress}
assert actual == expected
assert all(r['status'] == 'PASS' and r['mismatch_count'] == r['nonfinite'] == 0 for r in stress)
report = dict(status='PASS', full_paired_reference_checks=28, stress_paired_reference_checks=48,
              candidate_reference_checks=38, source_hashes=hashes, unchanged_cuda_source_cases=unchanged,
              limitation='Counts are assertions, not distinct shapes. Not proof of all possible inputs, OJ SafeExecutor acceptance, device-object equality or 88 points.')
(dest/'nsa108_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report))
