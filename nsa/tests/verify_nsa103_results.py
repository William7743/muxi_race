"""Verify recorded local coverage, source identity, and unchanged-path fingerprints."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
dest = root / 'results/2026-09-09'
def rows(filename):
    return [json.loads(line) for line in (dest / filename).read_text().splitlines() if line.strip()]
full, stress = rows('nsa103_full.jsonl'), rows('nsa103_stress.jsonl')
assert len(full) == 28 and len(stress) == 32
names = {'submissions/nsa097.py', 'submissions/nsa103.py'}
expected_hash = {
    'submissions/nsa097.py': hashlib.sha256((root / 'probes/probe_nsa097_v318_standard_jit.py').read_text(encoding='utf-8').encode()).hexdigest(),
    'submissions/nsa103.py': hashlib.sha256((root / 'probes/probe_nsa103_h2_small_scalar_output.py').read_text(encoding='utf-8').encode()).hexdigest()}
unchanged = []
for cid in range(1, 15):
    group = {r['module']: r for r in full if r['case_id'] == cid}
    assert set(group) == names
    assert all(r['correct'] and r['seed'] == 42 and r['mode'] == 'public' for r in group.values())
    if cid != 13:
        assert len({r['kernel_source_sha256'] for r in group.values()}) == 1
        unchanged.append(cid)
    else:
        assert len({r['kernel_source_sha256'] for r in group.values()}) == 2
for r in full + stress:
    assert r['source_sha256'] == expected_hash[r['module']]
expected = {(b,l,u,n) for b,l in ((1,64),(1,128),(1,256),(1,257),(1,512),(1,513),(2,256),(2,257))
            for u in range(2) for n in names}
assert {(r['case']['B'], r['case']['seq_len'], r['update'], r['module']) for r in stress} == expected
assert all(r['status'] == 'PASS' and r['mismatch_count'] == r['nonfinite'] == 0 for r in stress)
report = dict(status='PASS', full_reference_checks=len(full), extra_reference_checks=len(stress),
              candidate_reference_checks=sum(r['module'].endswith('/nsa103.py') for r in full + stress),
              unchanged_cuda_source_cases=unchanged, source_hashes=expected_hash,
              limitation='Generated source equality, not binary-object equality. Local checks do not prove OJ acceptance or 88 points.')
(dest / 'nsa103_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report))
