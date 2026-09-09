"""Bounded source isolation and coordinate audit for gather scalar conversions."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = ast.parse((root / 'probes/probe_nsa103_h2_small_scalar_output.py').read_text(encoding='utf-8'))
original_factory = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'nsa_gather')
configs = json.loads("{\"104\":{\"allocation\":\"            p_local = T.alloc_local([N // 8], T.float32)\\n            p_half = T.alloc_local([N // 8], T.float16)\\n            p_frag = T.alloc_fragment((G, N), T.float16)\\n            T.annotate_layout({p_frag: T.Fragment((G, N),\\n                forward_thread_fn=lambda g, n: (n // (N // 2)) * 64 + ((n % 16) // 4) * 16 + g,\\n                forward_index_fn=lambda g, n: ((n % (N // 2)) // 16) * 4 + n % 4)})\\n\",\"old\":\"            for g, n in T.Parallel(G, N):\\n                prob[g, n] = score[g, n] / sm[g]\",\"replacement\":\"            for g, n in T.Parallel(G, N):\\n                p_local[((n % (N // 2)) // 16) * 4 + n % 4] = score[g, n] / sm[g]\\n            for i in T.unroll(N // 8):\\n                p_half[i] = p_local[i]\\n            for g, n in T.Parallel(G, N):\\n                p_frag[g, n] = p_half[((n % (N // 2)) // 16) * 4 + n % 4]\\n            T.copy(p_frag, prob)\"},\"105\":{\"allocation\":\"            o_local = T.alloc_local([4], T.float32)\\n            o_half = T.alloc_local([4], T.float16)\\n            o_frag = T.alloc_fragment((G, KD), T.float16)\\n            if stage_output:\\n                T.annotate_layout({o_frag: T.Fragment((G, KD),\\n                    forward_thread_fn=lambda g, d: (d // 16) * 64 + ((d % 16) // 4) * 16 + g,\\n                    forward_index_fn=lambda g, d: d % 4)})\\n\",\"old\":\"                    T.copy(out, kv[0:G, :])\",\"replacement\":\"                    for g, d in T.Parallel(G, KD):\\n                        o_local[d % 4] = out[g, d]\\n                    for i in T.unroll(4):\\n                        o_half[i] = o_local[i]\\n                    for g, d in T.Parallel(G, KD):\\n                        o_frag[g, d] = o_half[d % 4]\\n                    T.copy(o_frag, kv[0:G, :])\"}}")
for version, filename in ((104, 'probe_nsa104_gather_scalar_probability.py'), (105, 'probe_nsa105_gather_scalar_output.py')):
    source = (root / 'probes' / filename).read_text(encoding='utf-8')
    parsed = ast.parse(source)
    name = 'nsa_gather_scalar' + str(version)
    factory = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == name)
    parsed.body.remove(factory)
    dispatcher = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == '_get_kernel')
    body = dispatcher.body[-2].body
    gate = next(n for n in body if isinstance(n, ast.If) and any(isinstance(x, ast.Name) and x.id == name for x in ast.walk(n)))
    expected_gate = ast.parse('if fn is nsa_gather and S in (2, 4) and D == 64 and groups == 16 and block_size == 16:\n    fn = ' + name).body[0]
    assert ast.dump(gate) == ast.dump(expected_gate)
    body.remove(gate)
    assert ast.dump(parsed) == ast.dump(parent), 'Unexpected changes to original factories/dispatch'
    config = configs[str(version)]
    text = ast.get_source_segment(source, factory)
    assert text.count(config['allocation']) == text.count(config['replacement']) == 1
    text = text.replace(config['allocation'], '').replace(config['replacement'], config['old']).replace('def '+name+'(', 'def nsa_gather(')
    restored = ast.parse(text).body[0]
    restored.decorator_list = factory.decorator_list
    assert ast.dump(restored) == ast.dump(original_factory), 'Unexpected math/decorator change'
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            ast.literal_eval(node.value)
    print(json.dumps(dict(version=version, audit='PASS', source_lf_sha256=hashlib.sha256(source.encode()).hexdigest())))
for n in (32, 64):
    coords = [(col // (n // 2) * 64 + col % 16 // 4 * 16 + g,
               col % (n // 2) // 16 * 4 + col % 4) for g in range(16) for col in range(n)]
    assert len(set(coords)) == 16*n
    assert set(coords) == {(tx, i) for tx in range(128) for i in range(n // 8)}
print('PASS: probability layouts bijective for N32/N64; output layout is the N32 case')
print('LIMIT: not actual OJ SafeExecutor or proof of GPU numerical correctness')
