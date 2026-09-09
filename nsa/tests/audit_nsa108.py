"""NSA108: two previously tested factories plus a selective shape dispatcher."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = ast.parse((root/'probes/probe_nsa103_h2_small_scalar_output.py').read_text(encoding='utf-8'))
source = (root/'probes/probe_nsa108_gather_selective_scalar.py').read_text(encoding='utf-8')
parsed = ast.parse(source)
for version, filename in ((104,'probe_nsa104_gather_scalar_probability.py'), (105,'probe_nsa105_gather_scalar_output.py')):
    name = f'nsa_gather_scalar{version}'
    expected = ast.parse((root/'probes'/filename).read_text(encoding='utf-8'))
    expected_fn = next(n for n in expected.body if isinstance(n,ast.FunctionDef) and n.name == name)
    actual = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name == name)
    assert ast.dump(actual) == ast.dump(expected_fn)
    parsed.body.remove(actual)
dispatch = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name == '_get_kernel')
body = dispatch.body[-2].body
gate = next(n for n in body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id == 'nsa_gather_scalar104' for x in ast.walk(n)))
expected_gate = ast.parse('if fn is nsa_gather and D == 64 and groups == 16 and block_size == 16:\n    if S == 2:\n        fn = nsa_gather_scalar104\n    elif S == 4 and seq_len >= 512:\n        fn = nsa_gather_scalar105').body[0]
assert ast.dump(gate) == ast.dump(expected_gate)
body.remove(gate)
assert ast.dump(parsed) == ast.dump(parent)
for node in ast.parse(source).body:
    if isinstance(node,ast.Assign):
        ast.literal_eval(node.value)
assert 'compile_flags=' not in source
print(json.dumps(dict(audit='PASS', source_lf_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     limitation='Static source equivalence, not OJ loader execution.')))
