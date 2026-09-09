"""Check S4 clones differ only by the explicit synchronization schedule."""
import ast
import copy
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent=ast.parse((root/"probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py").read_text(encoding="utf-8"))
base=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_gather")
def dump(n):
    return ast.dump(n,include_attributes=False)
def docs(t):
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return t
class StripSync(ast.NodeTransformer):
    def visit_Expr(self,n):
        if isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="T.sync_threads":
            assert not n.value.args and not n.value.keywords
            return None
        return self.generic_visit(n)
    def visit_If(self,n):
        n=self.generic_visit(n)
        if not n.body:
            assert not n.orelse
            assert ast.unparse(n.test) in ("conservative_reduce_fences","kd + 1 < D // KD","vd > 0")
            return None
        return n
outputs=[]
for v,file in ((134,"probe_nsa134_s4_manual_sync_conservative.py"),(135,"probe_nsa135_s4_manual_sync_reduce_fences.py")):
    path=root/"probes"/file
    tree=ast.parse(path.read_text(encoding="utf-8"))
    f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_gather_sync134")
    deco=copy.deepcopy(base.decorator_list)
    d=deco[0].keywords[0].value
    d.keys.append(ast.parse("tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC",mode="eval").body)
    d.values.append(ast.Constant(True))
    assert dump(f.decorator_list[0])==dump(deco[0])
    assert isinstance(f.body[0],ast.Assert) and ast.unparse(f.body[0].test)=="selected_blocks == 4 and dim == 64 and (groups == 16) and (block_size == 16)"
    flag=f.body[1]
    assert isinstance(flag,ast.Assign) and ast.unparse(flag.targets[0])=="conservative_reduce_fences"
    assert flag.value.value==(v==134)
    transformed=copy.deepcopy(f)
    transformed.body=transformed.body[2:]
    transformed=StripSync().visit(transformed)
    transformed.name="nsa_gather"
    transformed.decorator_list=copy.deepcopy(base.decorator_list)
    assert dump(transformed)==dump(base),"Non-synchronization math/body changed"
    tree.body.remove(f)
    dispatch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_get_kernel")
    condition="fn is nsa_gather and S == 4 and (D == 64) and (groups == 16) and (block_size == 16)"
    guard=next(n for n in ast.walk(dispatch) if isinstance(n,ast.If) and ast.unparse(n.test)==condition)
    assert len(guard.body)==1 and ast.unparse(guard.body[0])=="fn = nsa_gather_sync134"
    container=next(n for n in ast.walk(dispatch) if hasattr(n,"body") and isinstance(n.body,list) and guard in n.body)
    container.body.remove(guard)
    assert dump(docs(tree))==dump(docs(copy.deepcopy(parent))),"Other implementation or dispatch changed"
    outputs.append(dict(version=v,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status="PASS"))
print(json.dumps(dict(static_audit="PASS",versions=outputs,
    limitation="Does not establish barrier sufficiency; generated code and runtime verification are required.")))
