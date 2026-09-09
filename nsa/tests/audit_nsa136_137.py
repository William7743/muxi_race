"""Exact one-factor deltas from NSA134; no new math or input-dependent dispatch."""
import ast
import copy
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
parent=ast.parse((root/"probes/probe_nsa134_s4_manual_sync_conservative.py").read_text(encoding="utf-8"))
def normalize(tree):
    while isinstance(tree.body[0],ast.Expr) and isinstance(tree.body[0].value,ast.Constant) and isinstance(tree.body[0].value.value,str):
        tree.body.pop(0)
    return ast.dump(tree,include_attributes=False)
reports=[]
for v,file in ((136,"probe_nsa136_s4_manual_sync_safe_off.py"),(137,"probe_nsa137_s4_manual_sync_k_before_q.py")):
    tree=copy.deepcopy(parent)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_gather_sync134")
    if v==136:
        config=fn.decorator_list[0].keywords[0].value
        config.keys.append(ast.Constant("tl.disable_safe_memory_legalize"))
        config.values.append(ast.Constant(True))
    else:
        loop=next(n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.target)=="kd")
        assert ast.unparse(loop.body[0].target)=="(g, d)" and ast.unparse(loop.body[1].target)=="(n, d)"
        loop.body[0],loop.body[1]=loop.body[1],loop.body[0]
    path=root/"probes"/file
    actual=ast.parse(path.read_text(encoding="utf-8"))
    assert normalize(tree)==normalize(actual)
    reports.append(dict(version=v,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ast_delta="PASS"))
print(json.dumps(reports))

