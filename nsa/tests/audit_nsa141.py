"""Check unchanged parent code, exact output-coordinate coverage and scalar types."""
import ast
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
paths=[root/"probes/probe_nsa138_h2_bounded.py",root/"probes/probe_nsa141_d128_combined_output.py"]
trees=[ast.parse(p.read_text(encoding="utf-8")) for p in paths]
def dump(n): return ast.dump(n,include_attributes=False)
funcs=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="nsa_d128_scheduler") for t in trees]
assert dump(funcs[0].decorator_list[0])==dump(funcs[1].decorator_list[0])
for t,f in zip(trees,funcs):
    t.body=[n for n in t.body if n is not f and not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
assert dump(trees[0])==dump(trees[1])
coordinates=Counter((tx%16,cv*64+(tx//64)*32+j*16+(tx%64//16)*4+k)
    for tx in range(128) for cv in range(2) for j in range(2) for k in range(4))
assert set(coordinates)=={(i,j) for i in range(16) for j in range(128)}
assert set(coordinates.values())=={1}
source=paths[1].read_text(encoding="utf-8")
assert "saved_h = T.alloc_local([8], dtype)" in source
assert "saved_h[j] = local_h[j]" in source
assert "T.copy(Os_full, Output" in source
print(json.dumps(dict(status="PASS",source_sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest(),
    output_elements=len(coordinates),limitation="Coordinate bijection is not proof of inferred fragment layout or runtime precision.")))
