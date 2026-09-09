"""Verify probe isolation from NSA109; not a numerical or performance proof."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = (root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
candidate = (root/'probes/probe_nsa110_pair_scores_shared_reuse.py').read_text(encoding='utf-8')
gate = '        if fn is nsa_online_direct_output:\n            fn = nsa_online_pair_reuse110\n'
assert candidate.count(gate) == 1
tree = ast.parse(candidate.replace(gate,''))
new_functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'nsa_online_pair_reuse110']
assert len(new_functions) == 1
tree.body.remove(new_functions[0])
pt = ast.parse(parent)
for t in (tree,pt):
    while (isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant)
           and isinstance(t.body[0].value.value,str)):
        t.body.pop(0)
assert ast.dump(tree) == ast.dump(pt), 'Unexpected inherited code changes'
for n in ast.parse(candidate).body:
    if isinstance(n,ast.Assign):
        ast.literal_eval(n.value)
assert 'KV = T.alloc_shared([BS, D], dtype)' in ast.unparse(new_functions[0])
assert 'threads=64' in ast.unparse(new_functions[0])
print(json.dumps(dict(audit='PASS', source_lf_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                      limitation='Static source isolation only; OJ SafeExecutor and GPU behavior require separate checks.')))
