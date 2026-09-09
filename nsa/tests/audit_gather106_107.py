"""Verify only K/V shared layout and the shape-limited factory selection differ."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = ast.parse((root/'probes/probe_nsa103_h2_small_scalar_output.py').read_text(encoding='utf-8'))
old = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'nsa_gather')
for version, size in ((106,4),(107,16)):
    source = (root/f'probes/probe_nsa{version}_gather_kv_vec{size}.py').read_text(encoding='utf-8')
    parsed = ast.parse(source)
    name = f'nsa_gather_kv_vec{size}'
    new = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    parsed.body.remove(new)
    dispatch = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == '_get_kernel')
    body = dispatch.body[-2].body
    gate = next(n for n in body if isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(n)))
    expected = ast.parse('if fn is nsa_gather and S in (2, 4) and D == 64 and groups == 16 and block_size == 16:\n    fn = '+name).body[0]
    assert ast.dump(gate) == ast.dump(expected)
    body.remove(gate)
    assert ast.dump(parsed) == ast.dump(parent)
    text = ast.get_source_segment(source, new)
    annotation = f'            T.annotate_layout({{kv: make_mma_swizzle_layout(kv, vecSize={size})}})\n'
    assert text.count(annotation) == 1
    restored = ast.parse(text.replace(annotation, '').replace('def '+name+'(', 'def nsa_gather(')).body[0]
    restored.decorator_list = new.decorator_list
    assert ast.dump(restored) == ast.dump(old)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            ast.literal_eval(node.value)
    print(json.dumps(dict(version=version, source_isolation='PASS', source_lf_sha256=hashlib.sha256(source.encode()).hexdigest())))
print('LIMIT: layout correctness requires GPU reference checks; not OJ SafeExecutor validation')
