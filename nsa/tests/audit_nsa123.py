"""Verify NSA123 only adds one audited NSA122 factory and a restricted dispatch."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
def read(name):
    return (root / "probes" / name).read_text(encoding="utf-8")
parent = read("probe_nsa109_s2_scalar_probability.py")
wide = read("probe_nsa122_d64_block_bounds.py")
candidate = read("probe_nsa123_d64_plain_block_bounds.py")
d = wide.index("def nsa_simplified_bounded122(")
a = wide.rfind("@tilelang.jit", 0, d)
b = wide.index("\n    return kernel", d) + len("\n    return kernel")
clone = wide[a:b].replace("_bounded122", "_bounded123")
gate = (
    "        if fn is nsa_simplified and S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:\n"
    "            fn = nsa_simplified_bounded123\n"
)
expected = parent.replace("def _get_kernel(", clone + "\n\n\ndef _get_kernel(")
expected = expected.replace("        kernel = fn(\n", gate + "        kernel = fn(\n")
def tree(source):
    t = ast.parse(source)
    while isinstance(t.body[0], ast.Expr) and isinstance(t.body[0].value, ast.Constant) and isinstance(t.body[0].value.value, str):
        t.body.pop(0)
    return ast.dump(t)
assert tree(candidate) == tree(expected)
print(json.dumps(dict(status="PASS", source_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    changes="One bounded plain D64 factory; original dispatch otherwise unchanged")))
