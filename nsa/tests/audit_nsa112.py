"""Check NSA112's exact pair-reduction rewrite relative to NSA111."""
import ast
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parent = (root/'probes/probe_nsa111_pair_first_v_early.py').read_text(encoding='utf-8')
child = (root/'probes/probe_nsa112_pair_combined_reductions.py').read_text(encoding='utf-8')
edits = [
  [
    "            m0 = T.alloc_fragment([G], accum)\n            m1 = T.alloc_fragment([G], accum)\n",
    "            combined = T.alloc_fragment([G, BS], accum)\n"
  ],
  [
    "            sm1 = T.alloc_fragment([G], accum)\n",
    ""
  ],
  [
    "                    T.reduce_max(score0, m0, dim=1, clear=True)\n                    T.reduce_max(score1, m1, dim=1, clear=True)\n",
    "                    for g, n in T.Parallel(G, BS):\n                        combined[g, n] = T.max(score0[g, n], score1[g, n])\n                    T.reduce_max(combined, mx, dim=1, clear=True)\n"
  ],
  [
    "                        mx[g] = T.max(prev[g], T.max(m0[g], m1[g]))\n",
    "                        mx[g] = T.max(prev[g], mx[g])\n"
  ],
  [
    "                    T.reduce_sum(score0, sm0, dim=1)\n                    T.reduce_sum(score1, sm1, dim=1)\n",
    "                    for g, n in T.Parallel(G, BS):\n                        combined[g, n] = score0[g, n] + score1[g, n]\n                    T.reduce_sum(combined, sm0, dim=1)\n"
  ],
  [
    "                        ls[g] = ls[g] * sc[g] + sm0[g] + sm1[g]\n",
    "                        ls[g] = ls[g] * sc[g] + sm0[g]\n"
  ]
]
expected = parent
for old, new in edits:
    assert expected.count(old) == 1
    expected = expected.replace(old, new)
expected = expected.replace('nsa_online_pair_reuse111', 'nsa_online_pair_reuse112')
trees = [ast.parse(s) for s in (expected, child)]
for tree in trees:
    while isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        tree.body.pop(0)
assert ast.dump(trees[0]) == ast.dump(trees[1])
fn = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == 'nsa_online_pair_reuse112')
reductions = [n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith('reduce_')]
assert sorted(reductions) == ['reduce_max', 'reduce_sum']
print(json.dumps(dict(audit='PASS', source_lf_sha256=hashlib.sha256(child.encode()).hexdigest(),
                     limitation='Run NSA110/111 audits for inherited isolation; altered FP32 sum order needs runtime checking.')))
