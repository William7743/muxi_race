"""NSA109 is NSA104 with its dispatch narrowed to the reproducibly faster S2 path."""
import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
reference=(root/'probes/probe_nsa104_gather_scalar_probability.py').read_text(encoding='utf-8')
source=(root/'probes/probe_nsa109_s2_scalar_probability.py').read_text(encoding='utf-8')
old='if fn is nsa_gather and S in (2, 4) and D == 64 and groups == 16 and block_size == 16:'
new='if fn is nsa_gather and S == 2 and D == 64 and groups == 16 and block_size == 16:'
assert reference.count(old) == source.count(new) == 1
assert ast.dump(ast.parse(reference.replace(old,new))) == ast.dump(ast.parse(source))
for node in ast.parse(source).body:
    if isinstance(node,ast.Assign):
        ast.literal_eval(node.value)
print(json.dumps(dict(audit='PASS',source_lf_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     limitation='Run audit_gather104_105.py too for transitive isolation from NSA103. Not OJ SafeExecutor execution.')))
