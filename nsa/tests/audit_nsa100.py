"""Bounded NSA100 isolation audit; GPU correctness is a separate required gate."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = (root / 'probes/probe_nsa097_v318_standard_jit.py').read_text(encoding='utf-8')
candidate = (root / 'probes/probe_nsa100_d64_wide_v_fragment.py').read_text(encoding='utf-8')
before, after = ast.parse(parent), ast.parse(candidate)
name = 'nsa_simplified_k_fragment_v_packed'
new = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
after.body.remove(new)
dispatch = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == '_get_kernel')
body = dispatch.body[-2].body
gate = next(n for n in body if isinstance(n, ast.If) and any(
    isinstance(x, ast.Name) and x.id == name for x in ast.walk(n)))
assert ast.unparse(gate.test) == 'fn is nsa_simplified_k_fragment and D == 64 and (groups == 16) and (block_size == 16)'
body.remove(gate)
assert ast.dump(before) == ast.dump(after), 'Unexpected edits outside added factory and dispatch gate'
for node in ast.parse(candidate).body:
    if isinstance(node, ast.Assign):
        ast.literal_eval(node.value)
source = ast.get_source_segment(candidate, new)
assert 'T.copy(Vlinear, Vs)' in source
assert 'forward_thread_fn=lambda i, j: i * 4 + j // 16' in source
assert not any(x in source for x in ('call_extern', 'import_source', 'async', 'Pipelined'))
coords = [(i * 4 + j // 16, j % 16) for i in range(16) for j in range(64)]
assert len(set(coords)) == 1024
assert set(coords) == {(tx, k) for tx in range(64) for k in range(16)}
print(json.dumps({'isolation': 'PASS', 'load_layout_bijection': 'PASS',
                  'candidate_lf_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
                  'limitation': 'Not the OJ SafeExecutor; does not prove layout lowering or numerical correctness.'}))
