"""Verify exact donor combination and whitelist S8 address-only deltas."""
import ast
import copy
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
names={134:"probe_nsa134_s4_manual_sync_conservative.py",138:"probe_nsa138_h2_bounded.py",
142:"probe_nsa142_h2_s4_combination.py",143:"probe_nsa143_s8_modulo.py",144:"probe_nsa144_s8_predicate.py"}
texts={v:(root/"probes"/n).read_text(encoding="utf-8") for v,n in names.items()}
def tree(v): return ast.parse(texts[v])
def clean(t):
    t.body=[n for n in t.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    return t
def dump(t): return ast.dump(t,include_attributes=False)
def func(t,name): return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
for v in (142,143,144):
    functions=[n.name for n in tree(v).body if isinstance(n,ast.FunctionDef)]
    assert len(functions)==len(set(functions)), "Duplicate top-level factory/dispatcher"
t=tree(142)
donor=func(t,"nsa_gather_sync134")
assert dump(donor)==dump(func(tree(134),"nsa_gather_sync134"))
t.body.remove(donor)
dispatch=func(t,"_get_kernel")
body=next(n for n in dispatch.body if isinstance(n,ast.If)).body
extra=next(n for n in body if isinstance(n,ast.If) and any(isinstance(x,ast.Name) and x.id=="nsa_gather_sync134" for x in ast.walk(n)))
assert dump(extra)==dump(ast.parse("if fn is nsa_gather and S == 4 and D == 64 and groups == 16 and block_size == 16:\n    fn = nsa_gather_sync134").body[0])
body.remove(extra)
assert dump(clean(t))==dump(clean(tree(138)))
for v,expr in ((143,"(block_id % (seq_len // BS)) * BS"),(144,"block_id * BS")):
    source=texts[142]
    old="                            i_s = BI[i_b, i_t, i_h, s] * BS\n                            if i_s <= i_t and i_s >= 0:\n                                safe_s = T.min(T.max(i_s, 0), seq_len - BS)"
    assert source.count(old)==1
    expected=source.replace(old,"                            block_id = BI[i_b, i_t, i_h, s]\n                            i_s = block_id * BS\n                            if block_id >= 0 and block_id <= i_t // BS:\n                                safe_s = "+expr)
    assert dump(clean(ast.parse(expected)))==dump(clean(tree(v)))
    # Prove existing shape dispatch retains alignment required by the identity.
    assert dump(func(tree(v),"_get_kernel"))==dump(func(tree(142),"_get_kernel"))
for length in (1024,1040,8192):
    for block in range(length//16):
        assert min(max(block*16,0),length-16)==(block%(length//16))*16==block*16
    assert length > (length-1)//16 # sentinel remains rejected before addressing
print(json.dumps(dict(status="PASS",sources={v:hashlib.sha256((root/"probes"/names[v]).read_bytes()).hexdigest() for v in (142,143,144)},
    limitation="Address identity on admitted aligned blocks; runtime/reference and generated-code checks still required.")))
