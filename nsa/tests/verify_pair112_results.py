"""Verify existing NSA112 evidence; no GPU execution or operator generation."""
import hashlib
import json
from pathlib import Path
from reconcile_oj_calibration import qualified

root = Path(__file__).resolve().parents[1]
dest = root / 'results/2026-09-09'
paths = {
    'submissions/nsa109.py': 'probe_nsa109_s2_scalar_probability.py',
    'submissions/nsa110.py': 'probe_nsa110_pair_scores_shared_reuse.py',
    'submissions/nsa111.py': 'probe_nsa111_pair_first_v_early.py',
    'submissions/nsa112.py': 'probe_nsa112_pair_combined_reductions.py',
}
hashes = {m: hashlib.sha256((root/'probes'/p).read_text(encoding='utf-8').encode()).hexdigest()
          for m, p in paths.items()}

def read(name):
    return [json.loads(line) for line in (dest/name).read_text().splitlines() if line.strip()]

initial = read('nsa112.jsonl')
quiet = read('pairs_quiet_0856.jsonl')
stress = read('nsa112_stress.jsonl')
assert len(initial) == 6 and len(quiet) == 8 and len(stress) == 16
for rows, modules, seed, rounds in (
    (initial, set(paths)-{'submissions/nsa110.py'}, 211, 3),
    (quiet, set(paths), 137, 5),
):
    assert {(r['module'], r['case_id'], r['seed'], r['mode']) for r in rows} == {
        (m, 12, seed, mode) for m in modules for mode in ('public', 'current')}
    for r in rows:
        assert r['correct'] and r['source_sha256'] == hashes[r['module']]
        assert len(r['samples_us']['cupti']) == rounds
assert {(r['case']['seq_len'], r['case']['kv_heads'], r['update']) for r in stress} == {
    (length, heads, update) for length, heads in ((1024, 1), (1040, 1), (1025, 1), (1024, 2))
    for update in range(4)}
for r in stress:
    assert r['module'] == 'submissions/nsa112.py'
    assert r['source_sha256'] == hashes[r['module']]
    assert r['status'] == 'PASS' and r['nonfinite'] == r['mismatch_count'] == 0
    c = r['case']
    assert (c['B'], c['dim'], c['selected_blocks'], c['block_size'], c['causal']) == (1, 64, 8, 16, 1)
    assert c['q_heads'] == 16 * c['kv_heads']
resources = read('nsa112_resources.jsonl')
assert len(resources) == 2
assert {r['module'] for r in resources} == {'submissions/nsa111.py', 'submissions/nsa112.py'}
assert all(r['attributes_status'] == r['occupancy_status'] == 0 for r in resources)
report = dict(
    status='PASS', source_hashes=hashes, reference_assertions=len(initial)+len(quiet)+len(stress),
    candidate_reference_assertions=sum(r['module'] == 'submissions/nsa112.py' for r in initial+quiet+stress),
    initial_qualified_samples=sum(qualified(r)['accepted'] for r in initial), initial_samples=18,
    quiet_qualified_samples=sum(qualified(r)['accepted'] for r in quiet), quiet_samples=40,
    limitations=['Only target case12 timing; not full14-case certification.',
                 'Two case12 batches do not establish full-case performance or OJ improvement.',
                 'Resource query records lack source hashes; resource interpretation is corroborative only.'])
(dest/'pair112_verification.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report))
