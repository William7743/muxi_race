"""Verify controlled profiling metadata and summarize measured instruction counts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'results/2026-09-09/pair_profiles'
SOURCES = {'nsa109':'probe_nsa109_s2_scalar_probability.py',
           'nsa110':'probe_nsa110_pair_scores_shared_reuse.py',
           'nsa111':'probe_nsa111_pair_first_v_early.py'}
METRICS = {'Total Instructions','Compute Instructions','Memory Instructions',
           'Global Read Instructions','Global Write Instructions',
           'Private Read Instructions','Private Write Instructions'}
rows = []
for ver,file in SOURCES.items():
    meta = json.loads((DEST/f'profile_pair_{ver}_source.metadata.json').read_text())
    src = (ROOT/'probes'/file).read_text(encoding='utf-8')
    assert meta['source_sha256'] == hashlib.sha256(src.encode()).hexdigest()
    cuda = (DEST/f'profile_pair_{ver}_source.cu').read_text()
    normalized = '\n'.join(l.rstrip() for l in cuda.splitlines()).rstrip()+'\n'
    assert meta['kernel_source_sha256'] == hashlib.sha256(normalized.encode()).hexdigest()
    assert meta['correct'] and meta['case']['case_id'] == 12 and meta['seed'] == 0
    assert meta['mode'] == 'public' and meta['cold'] and meta['warmup'] == 10
    assert meta['cache_flush_bytes'] == 256000000 and meta['capture_calls'] == 1
    data = json.loads((DEST/f'profile_pair_{ver}/1_period0_dumped_result.json').read_text())
    metrics = {}
    for group in data.values():
        for item in group:
            assert not item['isError'], item
            assert item['name'] not in metrics
            metrics[item['name']] = item['data']
    assert set(metrics) == METRICS
    assert all(isinstance(v,(int,float)) and v >= 0 for v in metrics.values())
    assert metrics['Total Instructions'] == metrics['Compute Instructions'] + metrics['Memory Instructions']
    base = rows[0]['metrics'] if rows else metrics
    rows.append(dict(version=ver, source_sha256=meta['source_sha256'],
                     metrics=metrics, relative_to_nsa109={k:metrics[k]/base[k]-1 if base[k] else None for k in METRICS}))
report = dict(metadata_status='PASS', rows=rows, limitations=[
    'These are instruction counts, not elapsed time, byte counts or memory transactions.',
    'Counters use one captured call per metric pass; profiler may rerun its child for event batches.',
    'One metadata file per version records the final correctness check, not every replay.',
    'NSA110/111 global-write counts differ unexpectedly despite matching writeback structure; reliability unresolved.',
    'Metadata PASS checks provenance and schema, not hardware-counter isolation or accuracy.',
    'Use 1_period0 records only; report_dumped_result contains process-wide aggregates.',
    'Shared-device load prevents interpreting profiling duration as a qualified benchmark.'])
(DEST/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
