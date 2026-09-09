"""Summarize stored paired measurements, enforcing both predeclared guard gates.

No fitted time correction or OJ score estimate. Ratios compare each implementation
to the same-case control measured immediately before and after it.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('input')
p.add_argument('--parent', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
rows = [json.loads(line) for line in Path(a.input).read_text().splitlines() if line.strip()]
grouped = {}
for row in rows:
    assert row['correct'], row
    key = (row['case_id'], row['seed'], row['mode'])
    values, guards = row['samples_us']['cupti'], row['guard_samples']['cupti']
    assert len(values) == len(guards)
    good = [(v, v / ((g['before_us'] + g['after_us']) / 2))
            for v, g in zip(values, guards)
            if max(g['before_us'], g['after_us']) / min(g['before_us'], g['after_us']) - 1 <= .03
            and .92 <= g['reference_ratio'] <= 1.08]
    assert row['module'] not in grouped.setdefault(key, {}), 'Duplicate module/shape/seed/mode'
    grouped[key][row['module']] = dict(
        correct=True, qualified=len(good), total=len(values),
        raw_median_us=statistics.median(values),
        qualified_median_us=statistics.median([x[0] for x in good]) if len(good) >= 2 else None,
        guard_relative_median=statistics.median([x[1] for x in good]) if len(good) >= 2 else None)
pairs = []
for key, group in grouped.items():
    parent, candidate = group[a.parent], group[a.candidate]
    usable = min(parent['qualified'], candidate['qualified']) >= 2
    pairs.append(dict(case_id=key[0], seed=key[1], mode=key[2], parent=parent,
                      candidate=candidate, usable=usable,
                      candidate_relative_time=(candidate['guard_relative_median'] /
                          parent['guard_relative_median']) if usable else None))
ratios = [x['candidate_relative_time'] for x in pairs if x['usable']]
report = dict(input=Path(a.input).name, parent=a.parent, candidate=a.candidate,
              reference_checks=len(rows), qualified_samples=sum(
                  x['qualified'] for group in grouped.values() for x in group.values()),
              total_samples=sum(x['total'] for group in grouped.values() for x in group.values()),
              pairs=pairs, geometric_relative_time=math.exp(sum(map(math.log, ratios))/len(ratios)) if ratios else None,
              limitation='Subset local timings, not a score or a complete performance/correctness certification.')
Path(a.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
