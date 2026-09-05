import ast
import hashlib

import audit_v716_combine as audit


names = {
    717: "probe_v717_v716_e64_stage2_bfrag_only.py",
    718: "probe_v718_v716_e64_stage1_terminal_k_only.py",
}
dispatch = {717: "_get_stage2", 718: "_get_stage1"}
selected = {
    717: "_moe_stage2_fast_bfrag_prefetch",
    718: "_moe_stage1_prefetch_giu_merge",
}


for version, filename in names.items():
    audit.FILES[version] = audit.DATA / filename
    audit.SOURCES[version] = audit.FILES[version].read_text(encoding="utf-8")
    audit.TREES[version] = ast.parse(audit.SOURCES[version])
    audit.FUNCS[version] = {
        node.name: node for node in audit.TREES[version].body
        if isinstance(node, ast.FunctionDef)
    }
    assert set(audit.FUNCS[version]) == set(audit.FUNCS[716])
    for name in audit.FUNCS[716]:
        if name != dispatch[version]:
            assert ast.dump(audit.FUNCS[version][name]) == ast.dump(audit.FUNCS[716][name])
            assert audit.segment(version, name) == audit.segment(716, name)
    assert audit.other(version) == audit.other(716)
    restored = audit.SOURCES[version].replace(
        audit.segment(version, dispatch[version]), audit.segment(716, dispatch[version])
    )
    base = audit.SOURCES[716]
    assert restored[restored.index("import torch"):] == base[base.index("import torch"):]

    # Independently reconstruct the one allowed AST change and compare it to the candidate.
    allowed = ast.parse(audit.segment(716, dispatch[version])).body[0]
    if version == 717:
        matches = [node for node in ast.walk(allowed)
                   if isinstance(node, ast.Compare)
                   and isinstance(node.left, ast.Name) and node.left.id == "num_experts"]
        assert len(matches) == 1
        matches[0].ops = [ast.In()]
        matches[0].comparators = [ast.Tuple(elts=[ast.Constant(32), ast.Constant(64)], ctx=ast.Load())]
    else:
        matches = [node for node in ast.walk(allowed)
                   if isinstance(node, ast.Name) and node.id == "_moe_stage1_prefetch_giu_merge_v527"]
        assert len(matches) == 1
        matches[0].id = selected[version]
    assert ast.dump(allowed) == ast.dump(audit.FUNCS[version][dispatch[version]])

    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            actual = audit.check_mock(version, experts, dtype)
            expected = audit.check_mock(716, experts, dtype)
            if experts == 64:
                index = 1 if version == 717 else 0
                _, args, config = expected[index]
                expected[index] = (selected[version], args, config)
            assert actual == expected, (version, experts, dtype, actual, expected)
            for name, _, _ in actual:
                assert audit.segment(version, name) == audit.segment(716, name)
            print(f"v{version} E{experts:02} {dtype}: 2 fresh inputs x 2 launches, paths/args/passes PASS")

    print(f"v{version}: exact source unchanged except header and one dispatch choice PASS")
    print(f"v{version} SHA256 {hashlib.sha256(audit.FILES[version].read_bytes()).hexdigest()}")

# E64's existing shape uses the same block dimensions as the E32 source builder.
assert 7168 % 64 == 0 and 7168 // 64 == 112
assert 2048 % 128 == 0 and 2048 % 64 == 0
terminal = audit.segment(716, selected[718])
assert "T.use_swizzle(3 if num_experts == 32 else 2" in terminal
assert "if actual_rows > 0:" in terminal
assert "terminal_k = k_steps - 1" in terminal
assert "k_steps = T.ceildiv(hidden, bh1)" in terminal
assert "for k in range(k_steps - 1):" in terminal
print("E64 terminal-K static dimensional/empty-row/swizzle compatibility PASS; GPU compilation remains untested")
