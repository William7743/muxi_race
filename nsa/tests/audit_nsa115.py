"""Verify the sole scoped D128 row reciprocal rewrite."""
import ast, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
c=(root/'probes/probe_nsa115_d128_row_reciprocal.py').read_text(encoding='utf-8')
a=p.index('def nsa_d128_scheduler('); b=p.index('\n    return kernel',a)
old="                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = acc_s[i, j] / sm[i]"
new="                for i in T.Parallel(G):\n                    sm[i] = 1.0 / sm[i]\n                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = acc_s[i, j] * sm[i]"
assert p[a:b].count(old)==1
e=p[:a]+p[a:b].replace(old,new)+p[b:]
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(e)==tree(c)
print(json.dumps(dict(audit='PASS',source_sha256=hashlib.sha256(c.encode()).hexdigest(),
    limitation='Floating-point operation order changes; GPU reference test required')))

