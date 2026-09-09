"""Scoped compact exchange edit; reduction order needs FP32 reference tests."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'probes/probe_nsa119_d128_disjoint_reduce.py').read_text(encoding='utf-8')
c=(root/'probes/probe_nsa121_d128_compact_reduce.py').read_text(encoding='utf-8')
edits=[["T.alloc_shared([2, 128], accum)","T.alloc_shared([2, 32], accum)"],["                # Protect completed QK reads if storage rewriting aliases Ks.\n                T.sync_threads()\n                reduce_pair[0, tx] = row_m[0]\n                T.sync_threads()\n                row_m[0] = T.max(row_m[0], reduce_pair[0, tx ^ 64])\n                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff)))\n                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff)))","                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff)))\n                row_m[0] = T.max(row_m[0], T.shfl_sync(row_m[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff)))\n                # Protect completed QK reads before reusing Ks storage.\n                T.sync_threads()\n                if tx % 64 < 16:\n                    reduce_pair[0, (tx // 64) * 16 + tx % 16] = row_m[0]\n                T.sync_threads()\n                row_m[0] = T.max(row_m[0], reduce_pair[0, (1 - tx // 64) * 16 + tx % 16])"],["                # Max reads and sum writes use disjoint halves of this buffer.\n                reduce_pair[1, tx] = row_s[0]\n                T.sync_threads()\n                row_s[0] = row_s[0] + reduce_pair[1, tx ^ 64]\n                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff))\n                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff))","                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 32, width=64, mask=T.uint64(0xffffffffffffffff))\n                row_s[0] = row_s[0] + T.shfl_sync(row_s[0], (tx % 64) ^ 16, width=64, mask=T.uint64(0xffffffffffffffff))\n                # Max reads and sum writes remain in disjoint regions.\n                if tx % 64 < 16:\n                    reduce_pair[1, (tx // 64) * 16 + tx % 16] = row_s[0]\n                T.sync_threads()\n                row_s[0] = row_s[0] + reduce_pair[1, (1 - tx // 64) * 16 + tx % 16]"]]
for old,new in edits:
    assert p.count(old)==1
    p=p.replace(old,new)
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(p)==tree(c)
lanes=[{(tx%16,(tx//16)*4+j) for j in range(4)} for tx in range(128)]
for offset in (32,16):
    lanes=[s|lanes[tx^offset] for tx,s in enumerate(lanes)]
buf={tx//64*16+tx%16:lanes[tx] for tx in range(128) if tx%64<16}
assert len(buf)==32
for tx in range(128):
    result=lanes[tx]|buf[(1-tx//64)*16+tx%16]
    assert result=={(tx%16,j) for j in range(32)}
print(json.dumps(dict(audit='PASS',source_sha256=hashlib.sha256(c.encode()).hexdigest(),
    limitation='Sum order changes; need generated-source and GPU reference checks')))
