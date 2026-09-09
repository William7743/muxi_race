"""CPU-only whitelist, dispatch and bit-exact packed first-V mapping audit."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
names={128:"probe_nsa128_d64_bounds_s8_packed_prefetch.py",
130:"probe_nsa130_d128_bs32_block_bounds.py",
132:"probe_nsa132_d128_bounded_packed_v.py",
133:"probe_nsa133_d128_packed_v_control.py"}

def normalized(source):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    return ast.dump(tree,include_attributes=False)

def packed_parent(source):
    start=source.index("def nsa_d128_scheduler(")
    end=source.index("\n    return kernel",start)
    part=source[start:end]
    assert part.count("Vpf = T.alloc_fragment([BS, DV], dtype)")==1
    part=part.replace("Vpf = T.alloc_fragment([BS, DV], dtype)","Vbits = T.alloc_local([8], T.uint32)")
    a=part.index("                if prefetch_v:\n")
    b=part.index("                for ck_i",a)
    part=part[:a]+'''                if prefetch_v:
                    for r in T.unroll(2):
                        Vbits[T.Ramp(r * 4, 1, 4)] = T.reinterpret(V[i_b, i_s + tx // 8 + r * 16, i_h, T.Ramp((tx % 8) * 8, 1, 8)], "uint32x4")
'''+part[b:]
    assert part.count("                        T.copy(Vpf, Vs)")==1
    part=part.replace("                        T.copy(Vpf, Vs)",'''                        for r in T.unroll(2):
                            for c in T.vectorized(8):
                                Vs[tx // 8 + r * 16, (tx % 8) * 8 + c] = T.reinterpret(T.Cast(T.uint16, Vbits[r * 4 + c // 2] >> ((c % 2) * 16)), T.float16)''')
    return source[:start]+part+source[end:]

reports=[]
for parent,candidate in ((130,132),(128,133)):
    p=(root/"probes"/names[parent]).read_text(encoding="utf-8")
    c=(root/"probes"/names[candidate]).read_text(encoding="utf-8")
    assert normalized(packed_parent(p))==normalized(c)
    reports.append(dict(version=candidate,parent=parent,ast_whitelist="PASS",
        source_sha256=hashlib.sha256((root/"probes"/names[candidate]).read_bytes()).hexdigest()))

coverage=Counter((tx//8+r*16,tx%8*8+c) for tx in range(128) for r in range(2) for c in range(8))
assert set(coverage)=={(r,c) for r in range(32) for c in range(64)}
assert set(coverage.values())=={1}
for lo in range(65536):
    hi=lo^0xffff
    packed=lo|(hi<<16)
    assert (packed&0xffff)==lo and ((packed>>16)&0xffff)==hi
print(json.dumps(dict(status="PASS",reports=reports,covered_elements=len(coverage),
    half_patterns=65536,math_threads_sync_dispatch_passes="unchanged",
    gpu_validation="NOT_ASSESSED_BY_STATIC_AUDIT")))

