"""CPU source/evidence audit of peer343, retained byte-exact as NSA129.

Peer GPU runs remain attributed to the peer. This script launches no GPU work.
"""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parent = root / "probes/probe_nsa109_s2_scalar_probability.py"
candidate = root / "probes/probe_nsa129_oj109_s8_packed_prefetch.py"
combined = root / "probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py"

def tree(path):
    return ast.parse(path.read_text(encoding="utf-8"))

def without_doc(t):
    if (isinstance(t.body[0], ast.Expr)
            and isinstance(t.body[0].value, ast.Constant)
            and isinstance(t.body[0].value.value, str)):
        t.body.pop(0)
    return t

p, c, combo = map(tree, (parent, candidate, combined))
name = "nsa_online_direct_output"
packed = next(n for n in combo.body if isinstance(n, ast.FunctionDef) and n.name == name)
old = next(n for n in p.body if isinstance(n, ast.FunctionDef) and n.name == name)
p.body[p.body.index(old)] = packed
assert ast.dump(without_doc(p)) == ast.dump(without_doc(c)), "Unexpected change outside S8 factory"
digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert digest == "d91a4b0a7709c7a6a795d79a7b0c1b87d94861ee979a91ed875626cd788cdbd4", "Not byte-exact peer343"
assert hashlib.sha256(parent.read_bytes()).hexdigest() == "7abe284475520edf64999c583148cde45cc1c8f0a017503cc98068e7f33986fa"
assert "compile_flags" not in candidate.read_text(encoding="utf-8")
print(json.dumps(dict(source_audit="PASS", source_sha256=digest,
    equivalent_peer="v343", changed_factory=name,
    limitation="CPU source audit only; peer test coverage must be assessed separately")))
