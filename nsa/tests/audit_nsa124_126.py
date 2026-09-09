"""Exact AST deltas: wider plain bound dispatch, and two K-before-Q controls."""
import ast
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = {109:"probe_nsa109_s2_scalar_probability.py",122:"probe_nsa122_d64_block_bounds.py",
         123:"probe_nsa123_d64_plain_block_bounds.py",124:"probe_nsa124_plain_bounds_all_grids.py",
         125:"probe_nsa125_d64_bounded_k_before_q.py",126:"probe_nsa126_d64_k_before_q_control.py"}
sources = {v:(root/"probes"/f).read_text(encoding="utf-8") for v,f in files.items()}
def astcode(s):
    t = ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
oldgate = "        if fn is nsa_simplified and S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:"
assert sources[123].count(oldgate) == 1
expected = sources[123].replace(oldgate,oldgate.replace(" and B * seq_len * H >= 4096",""))
assert astcode(expected) == astcode(sources[124])
for ver in (125,126):
    source = sources[122 if ver == 125 else 109]
    clones = ""
    gate = "        if S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:\n"
    for i,name in enumerate(("nsa_simplified_v_prefetch","nsa_simplified_k_fragment_v_prefetch")):
        oldname = name+"_bounded122" if ver == 125 else name
        d = source.index("def "+oldname+"(")
        a = source.rfind("@tilelang.jit",0,d)
        b = source.index("\n    return kernel",d)+len("\n    return kernel")
        clone = source[a:b].replace("def "+oldname+"(","def "+name+"_ordered"+str(ver)+"(")
        q = "            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)\n"
        assert clone.count(q) == 1
        clone = clone.replace(q,"").replace("                for i, j in T.Parallel(G, BS):\n",
            "    "+q+"                for i, j in T.Parallel(G, BS):\n",1)
        clones += clone+"\n\n\n"
        gate += "            "+("elif" if i else "if")+" fn is "+name+":\n                fn = "+name+"_ordered"+str(ver)+"\n"
    expected = sources[123].replace("def _get_kernel(",clones+"def _get_kernel(")
    expected = expected.replace("        kernel = fn(\n",gate+"        kernel = fn(\n")
    assert astcode(expected) == astcode(sources[ver])
print(json.dumps(dict(status="PASS",hashes={v:hashlib.sha256(sources[v].encode()).hexdigest() for v in (124,125,126)},
    scope="No math/synchronization edits; only supported aligned dispatch and Q load placement")))
