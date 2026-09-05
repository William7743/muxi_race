import ast
import hashlib

from audit_v717_v718 import audit


version = 719
audit.FILES[version] = audit.DATA / "probe_v719_v718_e64_stage2_bfrag_only.py"
audit.SOURCES[version] = audit.FILES[version].read_text(encoding="utf-8")
audit.TREES[version] = ast.parse(audit.SOURCES[version])
audit.FUNCS[version] = {
    node.name: node for node in audit.TREES[version].body
    if isinstance(node, ast.FunctionDef)
}
assert set(audit.FUNCS[719]) == set(audit.FUNCS[718])
for name in audit.FUNCS[718]:
    if name != "_get_stage2":
        assert ast.dump(audit.FUNCS[719][name]) == ast.dump(audit.FUNCS[718][name]), name
        assert audit.segment(719, name) == audit.segment(718, name), name
assert audit.other(719) == audit.other(718)
assert audit.segment(719, "_get_stage2") == audit.segment(717, "_get_stage2")

allowed = ast.parse(audit.segment(718, "_get_stage2")).body[0]
matches = [node for node in ast.walk(allowed)
           if isinstance(node, ast.Compare)
           and isinstance(node.left, ast.Name) and node.left.id == "num_experts"]
assert len(matches) == 1
matches[0].ops = [ast.In()]
matches[0].comparators = [ast.Tuple(elts=[ast.Constant(32), ast.Constant(64)], ctx=ast.Load())]
assert ast.dump(allowed) == ast.dump(audit.FUNCS[719]["_get_stage2"])
restored = audit.SOURCES[719].replace(audit.segment(719, "_get_stage2"), audit.segment(718, "_get_stage2"))
base = audit.SOURCES[718]
assert restored[restored.index("import torch"):] == base[base.index("import torch"):]

for experts in (1, 8, 16, 32, 64):
    for dtype in ("float16", "float32"):
        actual = audit.check_mock(719, experts, dtype)
        expected = audit.check_mock(718, experts, dtype)
        if experts == 64:
            # E64 only: unchanged v718 Stage1 followed by the exact v717 Stage2.
            expected[1] = audit.check_mock(717, experts, dtype)[1]
        assert actual == expected, (experts, dtype, actual, expected)
        for name, _, _ in actual:
            assert audit.segment(719, name) == audit.segment(718, name)
        print(f"v719 E{experts:02} {dtype}: 2 fresh inputs x 2 launches, current inputs/paths/args/passes PASS")

print("v719 full-source/AST diff: header plus one E64 Stage2 dispatch condition only PASS")
print("E1/E8/E16/E32 full paths unchanged from v718; E64 uses v718 Stage1 + v717 Stage2 PASS")
print("All kernel builder bodies unchanged; no GPU/OJ test is implied by this audit")
print("v719 SHA256", hashlib.sha256(audit.FILES[719].read_bytes()).hexdigest())
