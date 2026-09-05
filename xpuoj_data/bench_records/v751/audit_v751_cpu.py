"""CPU-only exact composition and fresh-input host audit for v751."""

import ast
import copy
import hashlib
import itertools
import types
from pathlib import Path

D = Path(__file__).resolve().parents[2]
PATHS = {
    749: D / "probe_v749_v745_e16_stage1_runtime_m64.py",
    750: D / "probe_v750_v745_e16_stage2_runtime_m64.py",
    751: D / "probe_v751_v749_e16_stage2_runtime_m64.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
             for v, t in TREE.items()}


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


expected = copy.deepcopy(TREE[749])
for index, node in enumerate(expected.body):
    if isinstance(node, ast.FunctionDef) and node.name in {"_get_stage2", "run_kernel"}:
        expected.body[index] = copy.deepcopy(FUNCTIONS[750][node.name])
assert ast.dump(expected) == ast.dump(TREE[751])
text_expected = SOURCE[749][SOURCE[749].index("import torch\n"):]
for name in ("_get_stage2", "run_kernel"):
    old = ast.get_source_segment(SOURCE[749], FUNCTIONS[749][name])
    new = ast.get_source_segment(SOURCE[750], FUNCTIONS[750][name])
    assert text_expected.count(old) == 1
    text_expected = text_expected.replace(old, new)
assert text_expected == SOURCE[751][SOURCE[751].index("import torch\n"):]
print("Whole-module AST and executable text: exact v749 plus v750 Stage2/host PASS")

helper = ast.parse((D / "bench_records/v743/audit_v743_cpu.py").read_text(encoding="utf-8"))
nodes = [copy.deepcopy(n) for n in helper.body
         if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"Tensor", "host_mock"}]


class ExpectedEmptyScope(ast.NodeTransformer):
    count = 0

    def visit_Compare(self, node):
        if ast.unparse(node) == "experts == 32":
            self.count += 1
            return ast.parse("experts == 32 or (experts == 16 and hidden == 2048 and intermediate == 8192)",
                             mode="eval").body
        return self.generic_visit(node)


scope = ExpectedEmptyScope()
host_tree = ast.fix_missing_locations(scope.visit(module(nodes)))
assert scope.count == 2  # Only the oracle's padded-empty/raw-empty conditions.
ns = {"types": types, "copy": copy, "ast": ast, "module": module,
      "TREE": TREE, "FUNCTIONS": FUNCTIONS}
exec(compile(host_tree, "<expected-E16-empty-host>", "exec"), ns)
shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192),
          (16, 4096, 8192), (16, 2048, 4096), (32, 7168, 2048),
          (32, 2048, 8192), (64, 7168, 2048), (64, 2048, 8192))
count = 0
for (e, h, i), valid, padded, blocks, dtype in itertools.product(
        shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    base = ns["host_mock"](750, e, h, i, valid, padded, blocks, dtype)
    actual = ns["host_mock"](751, e, h, i, valid, padded, blocks, dtype)
    wanted = copy.deepcopy(base)
    if (e, h, i) == (16, 2048, 8192) and valid > 0 and padded > 0 and blocks > 0:
        assert wanted[0][0] == "_moe_stage1_prefetch"
        wanted[0] = ("_moe_stage1_e16_runtime_m64_prefetch", wanted[0][1])
    assert actual == wanted, (e, h, i, valid, padded, blocks, dtype)
    count += 1
print(f"Host {count} combinations x2 fresh inputs: v750 host plus exact v749 Stage1 PASS")
for v, path in PATHS.items():
    compile(SOURCE[v], str(path), "exec")
    print(f"v{v} SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
print("LIMIT: source/host composition, not GPU rounding, codegen, timing or malformed-metadata safety.")
