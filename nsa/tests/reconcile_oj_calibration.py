"""Audit local/OJ agreement without fitting candidate scores or selecting seeds.

Use a fixed, already submitted v236 anchor; NSA097/103 are holdout checks.
Checker text is authoritative for its score calculation, not alternate telemetry.
Generated CUDA equality is local evidence only, not OJ binary identity.
"""
import argparse
from decimal import Decimal, ROUND_FLOOR
import itertools
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results/2026-09-09'
MODULES = {141594: 'submissions/nsa_v236.py', 141647: 'submissions/nsa097.py',
           141648: 'submissions/nsa103.py', 141658: 'submissions/nsa109.py'}
NAMES = {141594: 'v236', 141647: 'NSA097', 141648: 'NSA103', 141658: 'NSA109'}
SOURCE_FILES = {
    141647: ROOT/'probes/probe_nsa097_v318_standard_jit.py',
    141648: ROOT/'probes/probe_nsa103_h2_small_scalar_output.py',
    141658: ROOT/'probes/probe_nsa109_s2_scalar_probability.py',
}


def score(base, elapsed):
    value = 100 * Decimal(str(base)) / (Decimal(str(base)) + Decimal(str(elapsed)))
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def checker_points(report):
    points = {}
    for row in report['cases']:
        if not row.get('result_available'):
            continue
        message = row.get('checkerMessage', '')
        base = re.search(r'Baseline:\s+([0-9.]+)\s+ms', message)
        elapsed = re.search(r'User kernel:\s+([0-9.]+)\s+ms', message)
        display = re.search(r'Display score:\s+(\d+)', message)
        config = dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', message))
        assert base and elapsed and display and config, 'Missing scored checker data'
        b, t = Decimal(base[1]) * 1000, Decimal(elapsed[1]) * 1000
        s = int(display[1])
        assert score(b, t) == s == row['displayScore']
        assert config['causal'] == 1
        cid = row['case_index']
        assert cid not in points
        telemetry = row.get('telemetry', {})
        tb = telemetry.get('tb_time_ms')
        points[cid] = dict(case_id=cid, baseline_us=float(b), oj_us=float(t),
                           score=s, config=config,
                           telemetry_baseline_disagrees=(tb is not None and abs(tb * 1000 - float(b)) > 1e-8))
    if report['meta']['status'] == 'Accepted':
        assert set(points) == set(range(1, 15))
        assert abs(sum(p['score'] for p in points.values()) / 14 - report['meta']['displayScore']) < .006
    return points


def qualified(row):
    samples, guards = row['samples_us']['cupti'], row['guard_samples']['cupti']
    assert len(samples) == len(guards)
    ratios, accepted = [], []
    for value, guard in zip(samples, guards):
        before, after = guard['before_us'], guard['after_us']
        assert value > 0 and before > 0 and after > 0
        drift = max(before, after) / min(before, after) - 1
        if drift <= .03 and .92 <= guard.get('reference_ratio', math.nan) <= 1.08:
            ratios.append(value / ((before + after) / 2))
            accepted.append(value)
    return dict(accepted=len(accepted), total=len(samples),
                local_us=stats.median(accepted) if len(accepted) >= 2 else None,
                guard_ratio=stats.median(ratios) if len(ratios) >= 2 else None)


def analyze(local_path):
    old = json.loads((DEST / 'oj_new_details.json').read_text(encoding='utf-8'))
    new = json.loads((DEST / 'oj_calibration_details.json').read_text(encoding='utf-8'))
    reports = {r['meta']['id']: r for r in old + new if r['meta']['id'] in MODULES}
    completed = {sid: r for sid, r in reports.items() if r['meta']['status'] == 'Accepted'}
    observed = {sid: checker_points(r) for sid, r in completed.items()}
    rows = [json.loads(l) for l in local_path.read_text().splitlines() if l.strip()]
    reference_rows = [json.loads(l) for l in (DEST/'nsa097.jsonl').read_text().splitlines() if l.strip()]
    frozen_anchor_hashes = {r['source_sha256'] for r in reference_rows if r['module'] == MODULES[141594]}
    frozen_guard_hashes = {r['guard_source_sha256'] for r in reference_rows}
    assert len(frozen_anchor_hashes) == len(frozen_guard_hashes) == 1
    assert len({(r['module'], r['case_id']) for r in rows}) == len(rows), 'Do not pool seeds/runs'
    indexed = {(r['module'], r['case_id']): r for r in rows}
    for row in rows:
        assert row['correct'] and row['seed'] == 0 and row['mode'] == 'public'
        assert row['guard_module'] == 'submissions/nsa_v159.py'
        assert row['guard_source_sha256'] in frozen_guard_hashes
    validations, details = [], []
    for sid, report in completed.items():
        module = MODULES[sid]
        if (module, 1) not in indexed:
            continue
        pp = []
        for cid in range(1, 15):
            row, anchor_row = indexed[module, cid], indexed[MODULES[141594], cid]
            if sid != 141594:
                assert row['source_sha256'] == report['source']['sha256'], 'Submitted file mismatch'
                assert hashlib.sha256(SOURCE_FILES[sid].read_text(encoding='utf-8').encode()).hexdigest() == row['source_sha256']
            else:
                assert row['source_sha256'] in frozen_anchor_hashes, 'Frozen anchor changed'
                # The older source is executable-AST matched, not exact-text matched.
                matches = json.loads((DEST / 'oj_source_matches.json').read_text())
                match = next(m for m in matches if m['submission_id'] == sid)
                assert any(p.replace('\\', '/') == module for p in match['executable_ast_matches'])
            truth, anchor = observed[sid][cid], observed[141594][cid]
            config = {k: v for k, v in row['case'].items() if k != 'case_id'}
            assert config == truth['config'] == anchor['config']
            assert truth['baseline_us'] == anchor['baseline_us'], 'Baseline formula changed'
            q, aq = qualified(row), qualified(anchor_row)
            identical = row['kernel_source_sha256'] == anchor_row['kernel_source_sha256']
            point = dict(case_id=cid, **q, oj_us=truth['oj_us'], actual_point_score=truth['score'],
                         baseline_us=truth['baseline_us'], same_local_cuda_as_anchor=identical)
            if q['guard_ratio'] is not None and aq['guard_ratio'] is not None:
                ratio = q['guard_ratio'] / aq['guard_ratio']
                predicted = anchor['oj_us'] * ratio
                # A same-source path cannot establish an algorithmic timing improvement.
                structural = anchor['oj_us'] * (1 if identical else ratio)
                point.update(measured_relative_time=ratio, projected_oj_us=predicted,
                             projected_score=score(truth['baseline_us'], predicted),
                             no_unchanged_path_gain_us=structural,
                             no_unchanged_path_gain_score=score(truth['baseline_us'], structural),
                             raw_local_score=score(truth['baseline_us'], q['local_us']),
                             projected_abs_time_error_pct=abs(predicted / truth['oj_us'] - 1) * 100)
            pp.append(point)
        usable = all('projected_score' in p for p in pp)
        total = sum(p['projected_score'] for p in pp) / 14 if usable else None
        validations.append(dict(submission_id=sid, version=NAMES[sid], actual_score=report['meta']['displayScore'],
                                anchor_self_check=sid == 141594, projected_score=total,
                                projected_minus_actual=total-report['meta']['displayScore'] if usable else None,
                                no_unchanged_path_gain_score=sum(p['no_unchanged_path_gain_score'] for p in pp)/14 if usable else None,
                                raw_local_score=sum(p['raw_local_score'] for p in pp)/14 if usable else None,
                                median_abs_time_error_pct=stats.median(p['projected_abs_time_error_pct'] for p in pp) if usable else None,
                                points=pp))
    for a, b in itertools.combinations([r['submission_id'] for r in validations], 2):
        same, changed = [], []
        for cid in range(1, 15):
            ar, br = indexed[MODULES[a], cid], indexed[MODULES[b], cid]
            ap, bp = observed[a][cid], observed[b][cid]
            same_cuda = ar['kernel_source_sha256'] == br['kernel_source_sha256']
            entry = dict(case_id=cid, a_us=ap['oj_us'], b_us=bp['oj_us'],
                         delta_integer_score=bp['score']-ap['score'],
                         oj_time_ratio=bp['oj_us']/ap['oj_us'],
                         same_local_generated_cuda=same_cuda,
                         large_oj_gap=abs(bp['oj_us']-ap['oj_us']) > max(1, .03*min(bp['oj_us'],ap['oj_us'])))
            (same if same_cuda else changed).append(entry)
        details.append(dict(a=a, b=b, unchanged_paths=same, changed_paths=changed,
                            total_score_delta=sum(p['delta_integer_score'] for p in same+changed)/14,
                            unchanged_path_score_delta=sum(p['delta_integer_score'] for p in same)/14,
                            changed_path_score_delta=sum(p['delta_integer_score'] for p in changed)/14))
    errors = [abs(v['projected_minus_actual']) for v in validations
              if not v['anchor_self_check'] and v['projected_minus_actual'] is not None]
    return dict(anchor_id=141594, validations=validations, pair_diagnostics=details,
                heldout_score_errors=dict(count=len(errors), mean_abs=stats.mean(errors),
                                          max_abs=max(errors), is_future_error_bound=False),
                pending_ids=[sid for sid,r in reports.items() if r['meta']['status'] == 'Pending'],
                reference_checks=len(rows), qualified_samples=sum(qualified(r)['accepted'] for r in rows),
                total_samples=sum(qualified(r)['total'] for r in rows),
                telemetry_checker_baseline_conflicts=sum(p['telemetry_baseline_disagrees'] for ps in observed.values() for p in ps.values()),
                limitations=['No fitted per-case offsets, seed selection, or altered operator results.',
                             'Fixed v236 anchor is not a holdout sample; new submissions test transfer.',
                             'Printed integer-microsecond OJ times do not reveal raw timing/rounding implementation.',
                             'Local CUDA equality does not certify identical OJ binaries or environments.',
                             'Unchanged-path OJ variation has unknown cause; do not automatically drop slow submissions.',
                             '16GB/25% sGPU does not establish full64GB performance equivalence.'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', type=Path, default=DEST/'calibration_097_103_236.jsonl')
    p.add_argument('--output', type=Path, default=DEST/'oj_calibration_reconciliation.json')
    args = p.parse_args()
    result = analyze(args.local)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('validations','pair_diagnostics')}))
    for v in result['validations']:
        print(json.dumps({k:x for k,x in v.items() if k != 'points'}))
