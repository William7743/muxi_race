"""Prove scoped layout edits and conservative synchronization controls."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
a=p.index('def nsa_d128_scheduler(');b=p.index('\n    return kernel',a)
body=p[a:b]
assert body.count('T.sync_warp(T.uint64(0xffffffffffffffff))')==2
control=body.replace('T.sync_warp(T.uint64(0xffffffffffffffff))','T.sync_threads()')
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
for v,vec,label in [(116,16,'v_swizzle16'),(117,4,'v_swizzle4'),(118,None,'block_barrier_control')]:
    f=control
    if vec:
        f=f.replace('            Vpf = T.alloc_fragment([BS, DV], dtype)',
            f'            T.annotate_layout({{Vs: make_mma_swizzle_layout(Vs, vecSize={vec})}})\n            Vpf = T.alloc_fragment([BS, DV], dtype)')
        coords={(row, ((col//vec)^((row//1)%min(16,64//vec)))*vec+col%vec)
                for row in range(32) for col in range(64)}
        assert coords=={(row,col) for row in range(32) for col in range(64)}
    c=(root/f'probes/probe_nsa{v}_d128_{label}.py').read_text(encoding='utf-8')
    assert tree(c)==tree(p[:a]+f+p[b:])
    print(json.dumps(dict(version=v,audit='PASS',source_sha256=hashlib.sha256(c.encode()).hexdigest(),
                         scope='Only D128 Vs layout and two full-block barrier substitutions',
                         limitation='Bijection is not a bank-conflict or compiler correctness proof')))
