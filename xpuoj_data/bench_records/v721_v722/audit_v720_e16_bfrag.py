import ast
import hashlib

from audit_v719_combine import audit


version = 720
audit.FILES[version] = audit.DATA / "probe_v720_v719_e16_stage2_bfrag_only.py"
audit.SOURCES[version] = audit.FILES[version].read_text(encoding="utf-8")
audit.TREES[version] = ast.parse(audit.SOURCES[version])
audit.FUNCS[version] = {
    node.name: node for node in audit.TREES[version].body
    if isinstance(node, ast.FunctionDef)
}
assert set(audit.FUNCS[720]) == set(audit.FUNCS[719])
for name in audit.FUNCS[719]:
    if name != "_get_stage2":
        assert ast.dump(audit.FUNCS[720][name]) == ast.dump(audit.FUNCS[719][name]), name
        assert audit.segment(720, name) == audit.segment(719, name), name
assert audit.other(720) == audit.other(719)

allowed = ast.parse(audit.segment(719, "_get_stage2")).body[0]
matches = [node for node in ast.walk(allowed)
           if isinstance(node, ast.Compare)
           and isinstance(node.left, ast.Name) and node.left.id == "num_experts"]
assert len(matches) == 1 and isinstance(matches[0].ops[0], ast.In)
assert ast.literal_eval(matches[0].comparators[0]) == (32, 64)
matches[0].comparators = [ast.Tuple(elts=[ast.Constant(16), ast.Constant(32), ast.Constant(64)], ctx=ast.Load())]
assert ast.dump(allowed) == ast.dump(audit.FUNCS[720]["_get_stage2"])
restored = audit.SOURCES[720].replace(audit.segment(720, "_get_stage2"), audit.segment(719, "_get_stage2"))
base = audit.SOURCES[719]
assert restored[restored.index("import torch"):] == base[base.index("import torch"):]

for experts in (1, 8, 16, 32, 64):
    for dtype in ("float16", "float32"):
        actual = audit.check_mock(720, experts, dtype)
        expected = audit.check_mock(719, experts, dtype)
        if experts == 16:
            _, args, config = expected[1]
            assert args[:3] == (2048, 8192, 16)
            expected[1] = ("_moe_stage2_fast_bfrag_prefetch", args, config)
        assert actual == expected, (experts, dtype, actual, expected)
        for name, _, _ in actual:
            assert audit.segment(720, name) == audit.segment(719, name)
        print(f"v720 E{experts:02} {dtype}: 2 fresh inputs x 2 launches, current inputs/paths/args/passes PASS")

print("v720 full-source/AST diff: header plus one E16 Stage2 dispatch condition only PASS")
print("E1/E8/E32/E64 full paths and every Stage1 unchanged from v719 PASS")
print("E16 mock uses problem dimensions H2048/I8192; all kernel builders unchanged")
print("No GPU/OJ test is implied by this CPU audit")
print("v720 SHA256", hashlib.sha256(audit.FILES[720].read_bytes()).hexdigest())
