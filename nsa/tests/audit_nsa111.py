"""Verify that NSA111 only moves the first V shared load before softmax."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = (root/'probes/probe_nsa110_pair_scores_shared_reuse.py').read_text(encoding='utf-8')
child = (root/'probes/probe_nsa111_pair_first_v_early.py').read_text(encoding='utf-8')
load = '''                    if s0 >= 0 and s0 <= i_t:
                        safe_v0 = T.min(T.max(s0, 0), seq_len - BS)
                        for n, d in T.Parallel(BS, D):
                            KV[n, d] = V[i_b, safe_v0 + n, i_h, d]
'''
assert parent.count(load) == 1
expected = parent.replace(load, '                    if s0 >= 0 and s0 <= i_t:\n')
expected = expected.replace('                    T.copy(mx, prev)\n',load+'                    T.copy(mx, prev)\n')
expected = expected.replace('nsa_online_pair_reuse110','nsa_online_pair_reuse111')
trees = [ast.parse(s) for s in (child,expected)]
for tree in trees:
    while isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        tree.body.pop(0)
assert ast.dump(trees[0]) == ast.dump(trees[1])
print(json.dumps(dict(audit='PASS', source_lf_sha256=hashlib.sha256(child.encode()).hexdigest(),
                      limitation='Run audit_nsa110.py too. Source isolation is not runtime validation.')))
