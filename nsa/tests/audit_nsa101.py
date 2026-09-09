"""Static isolation and bit-exact host simulation of the packed V shuffle."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = (root / 'probes/probe_nsa097_v318_standard_jit.py').read_text(encoding='utf-8')
source = (root / 'probes/probe_nsa101_d64_packed_v_shuffle.py').read_text(encoding='utf-8')
before, after = ast.parse(parent), ast.parse(source)
name = 'nsa_simplified_k_fragment_v_shuffle'
new = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
after.body.remove(new)
dispatch = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == '_get_kernel')
body = dispatch.body[-2].body
gate = next(n for n in body if isinstance(n, ast.If) and any(
    isinstance(x, ast.Name) and x.id == name for x in ast.walk(n)))
assert ast.unparse(gate.test) == 'fn is nsa_simplified_k_fragment and D == 64 and (groups == 16) and (block_size == 16)'
body.remove(gate)
assert ast.dump(before) == ast.dump(after)
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign):
        ast.literal_eval(node.value)
factory = ast.get_source_segment(source, new)
assert not any(x in factory for x in ('call_extern', 'import_source', 'async', 'Pipelined'))
# Each input is a distinct uint16 bit pattern; simulate half2 packing without FP conversion.
registers = [[(tx // 16 * 4 + tx % 16 // 4) * 64 + cb * 16 + tx % 4 * 4 + c
              for cb in range(4) for c in range(4)] for tx in range(64)]
packed = [[values[p * 2] | (values[p * 2 + 1] << 16) for p in range(8)] for values in registers]
count = 0
for tx in range(64):
    for cb in range(4):
        for r in range(4):
            src_lane = tx // 16 * 16 + r * 4 + tx % 16 // 4
            lo, hi = packed[src_lane][cb * 2:cb * 2 + 2]
            word = lo if tx % 4 < 2 else hi
            actual = (word >> (tx % 2 * 16)) & 0xffff
            expected = (tx // 16 * 4 + r) * 64 + cb * 16 + tx % 16
            assert actual == expected, (tx, cb, r, actual, expected)
            count += 1
print(json.dumps({'isolation': 'PASS', 'simulated_half_values': count,
                  'candidate_lf_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'limitation': 'Simulation assumes the observed MACA MMA B layout. GPU reference checks and generated code still required.'}))
fullmask = (root / 'probes/probe_nsa102_d64_v_shuffle_fullmask.py').read_text(encoding='utf-8')
expected_fullmask = source.replace('src_lane, width=64)',
    'src_lane, width=64, mask=T.uint64(0xffffffffffffffff))')
assert ast.dump(ast.parse(fullmask)) == ast.dump(ast.parse(expected_fullmask))
assert fullmask.count('src_lane, width=64, mask=T.uint64(0xffffffffffffffff))') == 2
print('PASS: NSA102 differs only by two full 64-lane shuffle masks (comments ignored)')
