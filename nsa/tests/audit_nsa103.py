"""NSA103 is exactly NSA097 plus one previously tested factory and shape gate."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(name):
    return (root / 'probes' / name).read_text(encoding='utf-8')
source = read('probe_nsa103_h2_small_scalar_output.py')
parent = ast.parse(read('probe_nsa097_v318_standard_jit.py'))
reference = ast.parse(read('probe_nsa099_d64_scalar_output_lanes.py'))
candidate = ast.parse(source)
name = 'nsa_simplified_h2_scalar_output'
new = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == name)
candidate.body.remove(new)
expected = next(n for n in reference.body if isinstance(n, ast.FunctionDef) and n.name == 'nsa_simplified')
expected.name = name
assert ast.dump(new) == ast.dump(expected), 'Factory differs from validated NSA099 scalar output'
dispatch = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == '_get_kernel')
body = dispatch.body[-2].body
gate = next(n for n in body if isinstance(n, ast.If) and any(
    isinstance(x, ast.Name) and x.id == name for x in ast.walk(n)))
expected_gate = ast.parse('if fn is nsa_simplified and H == 2 and D == 64 and groups == 16 and block_size == 16 and B * seq_len * H <= 1024:\n    fn = nsa_simplified_h2_scalar_output').body[0]
assert ast.dump(gate) == ast.dump(expected_gate)
body.remove(gate)
assert ast.dump(parent) == ast.dump(candidate), 'Unexpected edit outside factory and dispatch gate'
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign):
        ast.literal_eval(node.value)
assert 'compile_flags=' not in source
print(json.dumps({'factory_and_dispatch_isolation': 'PASS',
                  'source_lf_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'limitation': 'Static audit only; not execution of OJ SafeExecutor.'}))
