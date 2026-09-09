"""Verify the combination is NSA125 with only the plain-grid threshold changed."""
import ast, hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
parent = (root/"probes/probe_nsa125_d64_bounded_k_before_q.py").read_text(encoding="utf-8")
candidate = (root/"probes/probe_nsa127_combined_d64_bounds.py").read_text(encoding="utf-8")
old = "        if fn is nsa_simplified and S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:"
assert parent.count(old) == 1
expected = parent.replace(old,old.replace(">= 4096",">= 1024"))
def tree(s):
    t = ast.parse(s)
    while isinstance(t.body[0],ast.Expr) and isinstance(t.body[0].value,ast.Constant) and isinstance(t.body[0].value.value,str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(expected) == tree(candidate)
print(json.dumps(dict(status="PASS",source_sha256=hashlib.sha256(candidate.encode()).hexdigest())))
