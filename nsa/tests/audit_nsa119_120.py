"""Exact D128 softmax edit and two-warp reduction routing audit."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
old="                T.reduce_max(acc_s, mx, dim=1, clear=True)\n                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)\n                T.reduce_sum(acc_s, sm, dim=1)\n                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = acc_s[i, j] / sm[i]"
new="                for i, j in T.Parallel(G, BS):\n                    local_p[j % 4] = acc_s[i, j]\n                row_m[0] = T.max(T.max(T.max(-T.infinity(accum), local_p[0]), local_p[1]), T.max(local_p[2], local_p[3]))\n                # Protect completed QK reads if storage rewriting aliases Ks.\n                T.sync_threads()\n                reduce_pair[0, tx] = row_m[0]\n                T.sync_threads()\n                row_m[0] = T.max(row_m[0], reduce_pair[0, tx ^ 64])\n                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff)))\n                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff)))\n                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = T.exp2((acc_s[i, j] - row_m[0]) * scale)\n                for i, j in T.Parallel(G, BS):\n                    local_p[j % 4] = acc_s[i, j]\n                row_s[0] = 0.0\n                for j in T.unroll(4):\n                    row_s[0] = row_s[0] + local_p[j]\n                # Max reads and sum writes use disjoint halves of this buffer.\n                reduce_pair[1, tx] = row_s[0]\n                T.sync_threads()\n                row_s[0] = row_s[0] + reduce_pair[1, tx ^ 64]\n                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff))\n                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff))\n                for i, j in T.Parallel(G, BS):\n                    acc_s[i, j] = acc_s[i, j] / row_s[0]"
allocold="            mx = T.alloc_fragment([G], accum)\n            sm = T.alloc_fragment([G], accum)"
allocnew="            reduce_pair = T.alloc_shared([2, 128], accum)\n            row_m = T.alloc_local([1], accum)\n            row_s = T.alloc_local([1], accum)"
a=p.index('def nsa_d128_scheduler(');b=p.index('\n    return kernel',a)
assert p[a:b].count(old)==p[a:b].count(allocold)==1
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
for v,label,control in [(119,'disjoint_reduce',False),(120,'disjoint_reduce_barrier_control',True)]:
    rep=new
    if control:
        rep=rep.replace('                reduce_pair[1, tx] = row_s[0]',
                        '                T.sync_threads()\n                reduce_pair[1, tx] = row_s[0]')
    expected=p[:a]+p[a:b].replace(allocold,allocnew).replace(old,rep)+p[b:]
    c=(root/f'probes/probe_nsa{v}_d128_{label}.py').read_text(encoding='utf-8')
    assert tree(c)==tree(expected)
    print(json.dumps(dict(version=v,audit='PASS',source_sha256=hashlib.sha256(c.encode()).hexdigest())))
# Each lane owns four adjacent score columns and its row is lane%16.
lanes=[{(tx%16,(tx//16)*4+j) for j in range(4)} for tx in range(128)]
for offset in (64,32,16):
    lanes=[s | lanes[tx^offset] for tx,s in enumerate(lanes)]
for tx,s in enumerate(lanes):
    assert s=={(tx%16,j) for j in range(32)}
assert set(range(128)).isdisjoint(range(128,256))
print('ROUTING_PASS: every row includes32 scores; shared max/sum regions disjoint.')
print('LIMITATION: inspect generated aliasing and barriers and run GPU reference before promotion.')
