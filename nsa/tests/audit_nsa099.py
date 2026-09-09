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
allocation = '\n            local_o = T.alloc_local([16], accum)\n            half_o = T.alloc_local([16], dtype)\n            tx = T.get_thread_binding(0)'
replacement = '''                if D == 64 and BS == 16 and G == 16:
                    for i, j in T.Parallel(G, D):
                        local_o[(j // 16) * 4 + j % 4] = acc_o[i, j]
                    for j in T.unroll(16):
                        half_o[j] = local_o[j]
                    for j in T.unroll(4):
                        for k in T.vectorized(4):
                            Os[tx % 16, j * 16 + (tx // 16) * 4 + k] = half_o[j * 4 + k]
                else:
                    T.copy(acc_o, Os)'''
names = {'nsa_simplified', 'nsa_simplified_k_fragment',
         'nsa_simplified_v_prefetch', 'nsa_simplified_k_fragment_v_prefetch'}
before, after = ast.parse(baseline), ast.parse(candidate)
assert len(before.body) == len(after.body)
for old, new in zip(before.body[1:], after.body[1:]):
    if isinstance(old, ast.FunctionDef) and old.name in names:
        text = ast.get_source_segment(candidate, new)
        assert text.count(allocation) == text.count(replacement) == 1
        restored = text.replace(allocation, '').replace(replacement, '                T.copy(acc_o, Os)')
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


addresses = [(tx % 16, j * 16 + (tx // 16) * 4 + k) for tx in range(64) for j in range(4) for k in range(4)]
assert len(addresses) == len(set(addresses)) == 1024
assert set(addresses) == {(i, j) for i in range(16) for j in range(64)}
print('PASS: explicit shared-store coordinates cover G16 D64 exactly once')
