"""Exact scoped edit audit; static mapping does not replace GPU validation."""
import ast
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parent = (root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
child = (root/'probes/probe_nsa114_d128_direct_output.py').read_text(encoding='utf-8')
old = "                    T.sync_warp(T.uint64(0xffffffffffffffff))\n                    for j in T.unroll(2):\n                        for k in T.vectorized(4):\n                            Vs[tx % 16, (tx // 64) * 32 + j * 16 + (tx % 64 // 16) * 4 + k] = local_h[j * 4 + k]\n                    T.sync_threads()\n                    T.copy(\n                        Vs[0:G, :],\n                        Output[i_b, i_t, i_h * G : (i_h + 1) * G, cv_i * DV : (cv_i + 1) * DV],\n                    )\n                    if cv_i + 1 < CV:\n                        T.sync_warp(T.uint64(0xffffffffffffffff))"
new = "                    for j in T.unroll(2):\n                        for k in T.vectorized(4):\n                            Output[i_b, i_t, i_h * G + tx % 16, cv_i * DV + (tx // 64) * 32 + j * 16 + (tx % 64 // 16) * 4 + k] = local_h[j * 4 + k]\n                    if cv_i + 1 < CV:\n                        # Both warps must finish reading Vs before its next refill.\n                        T.sync_threads()"
a = parent.index('def nsa_d128_scheduler(')
b = parent.index('\n    return kernel', a)
assert parent[a:b].count(old) == 1
expected = parent[:a] + parent[a:b].replace(old,new) + parent[b:]
def tree(s):
    t = ast.parse(s)
    while isinstance(t.body[0], ast.Expr) and isinstance(t.body[0].value, ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(child) == tree(expected)
coords = [(tx%16,(tx//64)*32+j*16+(tx%64//16)*4+k)
          for tx in range(128) for j in range(2) for k in range(4)]
assert len(coords) == len(set(coords)) == 16*64
assert set(coords) == {(i,j) for i in range(16) for j in range(64)}
print(json.dumps(dict(audit='PASS', source_sha256=hashlib.sha256(child.encode()).hexdigest(),
    scope='Only nsa_d128_scheduler epilogue changes; bijective 16x64 store mapping',
    limitation='Requires runtime correctness and race/performance checks')))

