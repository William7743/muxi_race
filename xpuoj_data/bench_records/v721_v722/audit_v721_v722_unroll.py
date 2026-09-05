import ast
import copy
import hashlib

from audit_v720_e16_bfrag import audit


ORIGINAL = "_moe_stage1_prefetch_giu_merge"
VERSIONS = {721: 2, 722: 3}


class ReplaceK(ast.NodeTransformer):
    def __init__(self, replacement):
        self.replacement = replacement

    def visit_Name(self, node):
        if node.id == "k":
            assert isinstance(node.ctx, ast.Load)
            return copy.deepcopy(self.replacement)
        return node


class ReplaceSteady(ast.NodeTransformer):
    def __init__(self, factor):
        self.factor = factor
        self.replaced = 0

    def visit_For(self, node):
        if ast.unparse(node.iter) != "range(k_steps - 1)":
            return self.generic_visit(node)
        self.replaced += 1
        assert node.target.id == "k" and not node.orelse
        expanded = []
        for offset in range(self.factor):
            expression = f"outer * {self.factor}" + (f" + {offset}" if offset else "")
            replacement = ast.parse(expression, mode="eval").body
            expanded.extend(ReplaceK(replacement).visit(copy.deepcopy(item)) for item in node.body)
        outer = copy.deepcopy(node)
        outer.target.id = "outer"
        outer.iter = ast.parse(f"range((k_steps - 1) // {self.factor})", mode="eval").body
        outer.body = expanded
        remainder = copy.deepcopy(node)
        remainder.iter = ast.parse(
            f"range(((k_steps - 1) // {self.factor}) * {self.factor}, k_steps - 1)", mode="eval"
        ).body
        return [outer, remainder]


for version, factor in VERSIONS.items():
    filename = f"probe_v{version}_v720_e32_stage1_unroll{factor}.py"
    new_name = f"{ORIGINAL}_unroll{factor}"
    audit.FILES[version] = audit.DATA / filename
    audit.SOURCES[version] = audit.FILES[version].read_text(encoding="utf-8")
    audit.TREES[version] = ast.parse(audit.SOURCES[version])
    audit.FUNCS[version] = {
        node.name: node for node in audit.TREES[version].body if isinstance(node, ast.FunctionDef)
    }
    assert set(audit.FUNCS[version]) == set(audit.FUNCS[720]) | {new_name}
    for name in audit.FUNCS[720]:
        if name != "_get_stage1":
            assert audit.segment(version, name) == audit.segment(720, name), (version, name)
            assert ast.dump(audit.FUNCS[version][name]) == ast.dump(audit.FUNCS[720][name])
    assert audit.other(version) == audit.other(720)

    # Independently reconstruct the permitted AST transformation. This checks every
    # load address, MMA order, barrier, pass configuration, allocation, guard and epilogue.
    transformer = ReplaceSteady(factor)
    expected = transformer.visit(copy.deepcopy(audit.FUNCS[720][ORIGINAL]))
    expected.name = new_name
    assert transformer.replaced == 1
    assert ast.dump(expected) == ast.dump(audit.FUNCS[version][new_name])
    dispatch = copy.deepcopy(audit.FUNCS[720]["_get_stage1"])
    branches = [node for node in ast.walk(dispatch)
                if isinstance(node, ast.IfExp) and ast.unparse(node.test) == "num_experts == 32"]
    assert len(branches) == 1 and branches[0].body.id == ORIGINAL
    branches[0].body.id = new_name
    assert ast.dump(dispatch) == ast.dump(audit.FUNCS[version]["_get_stage1"])

    restored = audit.SOURCES[version].replace(audit.segment(version, new_name) + "\n\n\n", "")
    restored = restored.replace(audit.segment(version, "_get_stage1"), audit.segment(720, "_get_stage1"))
    base = audit.SOURCES[720]
    assert restored[restored.index("import torch"):] == base[base.index("import torch"):]

    # Enumerate the actual candidate AST's loop bounds and address arithmetic, not
    # merely an independently written sequence. Cover all positive counts up to 513.
    body = audit.FUNCS[version][new_name]
    outer = next(node for node in ast.walk(body)
                 if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                 and node.target.id == "outer")
    remainder = next(node for node in ast.walk(body)
                     if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                     and node.target.id == "k")
    def evaluate(node, **env):
        return eval(compile(ast.Expression(node), "<audit>", "eval"), {"range": range}, env)
    for count in range(1, 514):
        observed = []
        for outer_value in evaluate(outer.iter, k_steps=count):
            for statement in outer.body:
                if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                    continue
                call = statement.value
                if ast.unparse(call.func) != "T.copy" or not isinstance(call.args[0], ast.Subscript):
                    continue
                source = call.args[0]
                if ast.unparse(source.value) != "gate_w":
                    continue
                kslice = source.slice.elts[-1]
                lower = evaluate(kslice.lower, outer=outer_value, bh1=64)
                upper = evaluate(kslice.upper, outer=outer_value, bh1=64)
                assert upper - lower == 64 and lower % 64 == 0
                observed.append(lower // 64)
        observed.extend(evaluate(remainder.iter, k_steps=count))
        assert observed == list(range(count - 1)), (version, count, observed)
        assert observed + [count - 1] == list(range(count))
    print(f"v{version}: actual AST K intervals cover 0..K-1 exactly once for K=1..513 PASS")
    print(f"v{version}: E32 K112 -> {111 // factor} groups x {factor}, {111 % factor} remainder, terminal111")
    print(f"v{version}: builder AST differs only in permitted steady-loop unroll; guards/terminal/epilogue unchanged PASS")

    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            actual = audit.check_mock(version, experts, dtype)
            expected = audit.check_mock(720, experts, dtype)
            if experts == 32:
                _, args, config = expected[0]
                expected[0] = (new_name, args, config)
            assert actual == expected, (version, experts, dtype)
            print(f"v{version} E{experts:02} {dtype}: current-input forwarding, 2 calls x 2 launches PASS")
    print(f"v{version} SHA256 {hashlib.sha256(audit.FILES[version].read_bytes()).hexdigest()}")
print("All checks are static/CPU host-dispatch checks; no GPU compilation/correctness/performance claim")
