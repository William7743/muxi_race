"""Offline bounded source audit and paired timing summary; no OJ prediction."""
import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--baseline', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--measurements')
a = p.parse_args()
baseline, candidate = (Path(x).read_text(encoding='utf-8') for x in (a.baseline, a.candidate))
allocation = '\n            local_p = T.alloc_local([4], accum)\n            half_p = T.alloc_local([4], dtype)'
replacement = '''                if D == 64 and BS == 16 and G == 16:
                    for i, j in T.Parallel(G, BS):
                        local_p[j % 4] = acc_s[i, j]
                    for j in T.unroll(4):
                        half_p[j] = local_p[j]
                    for i, j in T.Parallel(G, BS):
                        acc_cast[i, j] = half_p[j % 4]
                else:
                    T.copy(acc_s, acc_cast)'''
names = {'nsa_simplified', 'nsa_simplified_k_fragment',
         'nsa_simplified_v_prefetch', 'nsa_simplified_k_fragment_v_prefetch'}
before, after = ast.parse(baseline), ast.parse(candidate)
assert len(before.body) == len(after.body)
for old, new in zip(before.body[1:], after.body[1:]):
    if isinstance(old, ast.FunctionDef) and old.name in names:
        text = ast.get_source_segment(candidate, new)
        assert text.count(allocation) == text.count(replacement) == 1
        restored = text.replace(allocation, '').replace(replacement, '                T.copy(acc_s, acc_cast)')
        restored_ast = ast.parse(restored).body[0]
        restored_ast.decorator_list = new.decorator_list
        assert ast.dump(old) == ast.dump(restored_ast), old.name
    else:
        assert ast.dump(old) == ast.dump(new), getattr(old, 'name', type(old).__name__)
print(json.dumps(dict(source_audit='PASS', changed_factories=sorted(names),
                     baseline_lf_sha256=hashlib.sha256(baseline.encode()).hexdigest(),
                     candidate_lf_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                     limitation='AST isolation only; requires GPU numerical and generated-code checks.')))

if a.measurements:
    rows = [json.loads(line) for line in Path(a.measurements).read_text().splitlines()]
    for row in rows:
        samples = row['samples_us']['cupti']
        guards = row['guard_samples']['cupti']
        assert len(samples) == len(guards) and row['correct']
        ratios = [v / ((g['before_us'] + g['after_us']) / 2)
                  for v, g in zip(samples, guards)
                  if max(g['before_us'], g['after_us']) / min(g['before_us'], g['after_us']) - 1 <= .03
                  and .92 <= g['reference_ratio'] <= 1.08]
        print(json.dumps(dict(case_id=row['case_id'], module=row['module'],
                              local_us=statistics.median(samples), accepted=len(ratios),
                              median_ratio=statistics.median(ratios) if len(ratios) >= 2 else None)))
