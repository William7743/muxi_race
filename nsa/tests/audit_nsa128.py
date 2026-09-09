"""Isolate the peer S8 prefetch delta and audit 64-lane bit/load mappings.

This does not replace GPU compilation, numerical stress, or race instrumentation.
"""
import ast
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/"probes/probe_nsa127_combined_d64_bounds.py").read_text(encoding="utf-8")
c=(root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_text(encoding="utf-8")
a=p.index("def nsa_online_direct_output(")
b=p.index("\n    return kernel",a)+len("\n    return kernel")
old=p[a:b]
edits=[
    ("            i_h = ibh % head_kv\n",
     "            i_h = ibh % head_kv\n            lane = T.get_thread_binding(0)\n            Vbits = T.alloc_local([8], T.uint32)\n"),
    ("                                safe_s = T.min(T.max(i_s, 0), seq_len - BS)\n",
     "                                safe_s = T.min(T.max(i_s, 0), seq_len - BS)\n                                for r in T.unroll(2):\n                                    Vbits[T.Ramp(r * 4, 1, 4)] = T.reinterpret(V[i_b, safe_s + lane // 8 + r * 8, i_h, T.Ramp((lane % 8) * 8, 1, 8)], \"uint32x4\")\n"),
    ("                                for i, j in T.Parallel(BS, D):\n                                    Ks[i, j] = V[i_b, safe_s + i, i_h, j]\n",
     "                                for r in T.unroll(2):\n                                    for c in T.vectorized(8):\n                                        Ks[lane // 8 + r * 8, (lane % 8) * 8 + c] = T.reinterpret(T.Cast(T.uint16, Vbits[r * 4 + c // 2] >> ((c % 2) * 16)), T.float16)\n"),
]
new=old
for before,after in edits:
    assert new.count(before)==1
    new=new.replace(before,after)
expected=p[:a]+new+p[b:]
def tree(s):
    t=ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(expected)==tree(c)
# All16x64 values loaded and stored once, with the same row/column mapping.
positions=[]
for lane in range(64):
    for r in range(2):
        for col in range(8):
            row=lane//8+r*8
            column=(lane%8)*8+col
            assert 0<=r*4+col//2<8
            positions.append((row,column))
assert len(positions)==len(set(positions))==16*64
assert set(positions)=={(i,j) for i in range(16) for j in range(64)}
# Exhaustive 16-bit recovery under the paired-half word convention.
for h in range(65536):
    other=h^0xffff
    word=h|(other<<16)
    assert (word&0xffff)==h and ((word>>16)&0xffff)==other
# Existing specialization and lane width must still delimit the new fixed layout.
assert "if (fn is nsa_online and S == 8 and H == 1 and groups == 16" in c
assert "and D == 64 and block_size == 16 and seq_len >= 1024 and seq_len % block_size == 0" in c
assert c.count("fn = nsa_online_direct_output")==1
assert "threads=64" in new
assert old.count("T.sync_warp")==new.count("T.sync_warp")
assert old.count("T.gemm")==new.count("T.gemm")
assert "compile_flags" not in c
report=dict(static_audit="PASS",gpu_validation="NOT_ASSESSED_BY_THIS_STATIC_AUDIT",
    source_sha256=hashlib.sha256(c.encode()).hexdigest(),
    changed_factory="nsa_online_direct_output",mapped_values=len(positions),
    limitation="Bit convention and logical mapping only; use verify_nsa128_results.py for the separate GPU evidence audit")
print(json.dumps(report))
