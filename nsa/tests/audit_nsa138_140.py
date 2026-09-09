"""AST whitelist for H2 bounded-load, V-prefetch and Q/K-order experiments."""
import ast
import copy
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent_text=(root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_text(encoding="utf-8")
p=ast.parse(parent_text)
def nodoc(t):
    t.body=[n for n in t.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    return t
def dump(t): return ast.dump(t,include_attributes=False)
results=[]
for v,label in ((138,"bounded"),(139,"prefetch"),(140,"order")):
    path=root/f"probes/probe_nsa{v}_h2_{label}.py"
    t=ast.parse(path.read_text(encoding="utf-8"))
    name=f"nsa_h2_{label}{v}"
    new=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
    original=next(n for n in p.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_simplified_h2_scalar_output")
    expected_text=ast.get_source_segment(parent_text,original).replace(original.name,name,1)
    old_offset="            i_s = BI[i_b, i_t, i_h, 0] * BS\n            if i_s <= i_t:"
    assert expected_text.count(old_offset)==1
    expected_text=expected_text.replace(old_offset,"            block_id = BI[i_b, i_t, i_h, 0]\n            if block_id >= 0 and block_id <= i_t // BS:\n                i_s = (block_id % (seq_len // BS)) * BS")
    for shared,input_name in (("Ks","K"),("Vs","V")):
        old=f"                if i_s >= 0 and i_s + BS <= seq_len:\n                    for i, j in T.Parallel(BS, D):\n                        {shared}[i, j] = {input_name}[i_b, i_s + i, i_h, j]\n                else:\n                    T.copy({input_name}[i_b, i_s : i_s + BS, i_h, :], {shared})"
        assert expected_text.count(old)==1
        expected_text=expected_text.replace(old,f"                for i, j in T.Parallel(BS, D):\n                    {shared}[i, j] = {input_name}[i_b, i_s + i, i_h, j]")
    if v==139:
        expected_text=expected_text.replace("            Vs = T.alloc_shared([BS, D], dtype)","            Vs = T.alloc_shared([BS, D], dtype)\n            Vpref = T.alloc_fragment([BS, D], dtype)")
        vloop="                for i, j in T.Parallel(BS, D):\n                    Vs[i, j] = V[i_b, i_s + i, i_h, j]"
        expected_text=expected_text.replace(vloop,"                T.copy(Vpref, Vs)")
        reduce="                T.reduce_max(acc_s, mx, dim=1, clear=True)"
        expected_text=expected_text.replace(reduce,vloop.replace("Vs[i, j]","Vpref[i, j]")+"\n"+reduce)
    if v==140:
        expected_text=expected_text.replace("reorder = dim == 64 and groups == 16 and block_size == 16 and batch * seq_len * head_kv >= 4096 and seq_len >= 1024 and seq_len % block_size == 0","reorder = True")
    expected_clone=ast.parse(expected_text).body[0]
    expected_clone.decorator_list=copy.deepcopy(original.decorator_list)
    assert dump(expected_clone)==dump(new), "Unexpected clone-body change"
    dispatch=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="_get_kernel")
    guard=next(n for n in dispatch.body if isinstance(n,ast.If))
    extra=next(n for n in guard.body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id==name for x in ast.walk(n)))
    expected=ast.parse(f"if fn is nsa_simplified_h2_scalar_output and seq_len % block_size == 0:\n    fn = {name}").body[0]
    assert dump(extra)==dump(expected)
    guard.body.remove(extra); t.body.remove(new)
    assert dump(nodoc(t))==dump(nodoc(copy.deepcopy(p)))
    # Mechanical transformation is confined to the new clone. Existing kernels
    # and fallback selection are exactly parent128, including all pass configs.
    assert dump(new.decorator_list[0])==dump(next(n for n in p.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_simplified_h2_scalar_output").decorator_list[0])
    assert not any(isinstance(n,ast.Attribute) and n.attr in ("call_extern","import_source","Pipelined") for n in ast.walk(new))
    results.append(dict(version=v,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parent_unchanged=True))
# For aligned lengths and a current/past block, modulo leaves its offset exact.
for length in (16,64,128,256,512,1024):
    for token in range(length):
        for block in range(token//16+1):
            assert (block % (length//16))*16 == block*16
print(json.dumps(dict(status="PASS",probes=results,limitation="Clone arithmetic/runtime correctness requires GPU reference tests.")))
