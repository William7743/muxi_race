"""CPU-only v755 exact Stage1 composition and actual-branch tag audit."""

import ast
import copy
import hashlib
import itertools
import types
from pathlib import Path

D = Path(__file__).resolve().parents[2]
PATHS = {748: D / "probe_v748_v747_e64_stage1_runtime_m64.py",
         755: D / "probe_v755_v748_e32_stage1_runtime_m32_m64.py"}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)} for v, t in TREE.items()}
BASE = "_moe_stage1_runtime_m64_giu_merge"
NEW = "_moe_stage1_e32_runtime_m32_m64_giu_merge"


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def dump(n):
    return ast.dump(n, include_attributes=False)


def kernel(n):
    return next(x for x in ast.walk(n) if isinstance(x, ast.With))


def assignment(nodes, name):
    return next(n for n in nodes if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == name)


class TinyNames(ast.NodeTransformer):
    def visit_Name(self, n):
        if n.id.startswith("tail_"):
            n.id = "tiny_" + n.id[5:]
        return n


expected = copy.deepcopy(TREE[748])
old = FUNCTIONS[748][BASE]
clone = copy.deepcopy(old)
clone.name = NEW
pos = clone.body.index(assignment(clone.body, "tail_m")) + 1
clone.body.insert(pos, ast.parse("tiny_m = 32").body[0])
ck = kernel(clone)
pos = ck.body.index(assignment(ck.body, "tail_up_local")) + 1
ck.body[pos:pos] = [TinyNames().visit(copy.deepcopy(assignment(ck.body, name)))
                   for name in ("tail_gate_local", "tail_up_local")]
branch = ck.body[-1]
tail = branch.orelse[0]
assert not tail.orelse
tiny_body = TinyNames().visit(module(copy.deepcopy(tail.body))).body
tail.test = ast.parse("actual_rows > tiny_m", mode="eval").body
tail.orelse = [ast.If(test=ast.parse("actual_rows > 0", mode="eval").body, body=tiny_body, orelse=[])]
pos = next(i for i, n in enumerate(expected.body) if isinstance(n, ast.FunctionDef) and n.name == "_pick_tiles")
expected.body.insert(pos, clone)


class SelectE32(ast.NodeTransformer):
    count = 0

    def visit_Name(self, n):
        if n.id == BASE:
            self.count += 1
            return ast.IfExp(test=ast.parse("num_experts == 32", mode="eval").body,
                             body=ast.Name(id=NEW, ctx=ast.Load()), orelse=n)
        return n


selector = SelectE32()
selector.visit(next(n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == "_get_stage1"))
assert selector.count == 1 and dump(expected) == dump(TREE[755])
new = FUNCTIONS[755][NEW]
branch = kernel(new).body[-1]
old_branch = kernel(old).body[-1]
assert dump(module(branch.body)) == dump(module(old_branch.body))
assert dump(module(branch.orelse[0].body)) == dump(module(old_branch.orelse[0].body))
for name, n in FUNCTIONS[748].items():
    if name == "_get_stage1":
        continue
    other = FUNCTIONS[755][name]
    starts = [min([x.lineno] + [d.lineno for d in x.decorator_list]) for x in (n, other)]
    assert SOURCE[748].splitlines()[starts[0]-1:n.end_lineno] == SOURCE[755].splitlines()[starts[1]-1:other.end_lineno], name
calls = [n for n in ast.walk(new) if isinstance(n, ast.Call)]
assert sum(ast.unparse(n.func) == "T.Kernel" for n in calls) == 1
assert [ast.unparse(n.args[0]) for n in calls if ast.unparse(n.func) == "T.alloc_shared"] == ["(bt1, bh1)", "(be1, bh1)"]
print("Whole AST reconstructed from748; old full/tail bodies, all original builders/host text unchanged except exact E32 getter PASS", flush=True)

# Extract only existing host-mock definitions; adapt the oracle's two empty
# conditions to the already-existing v748 E64 target. No old tests run.
old_audit = ast.parse((D / "bench_records/v743/audit_v743_cpu.py").read_text(encoding="utf-8"))
defs = [copy.deepcopy(n) for n in old_audit.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in {"Tensor", "host_mock"}]


class EmptyScope(ast.NodeTransformer):
    count = 0

    def visit_Compare(self, n):
        if ast.unparse(n) == "experts == 32":
            self.count += 1
            return ast.parse("experts == 32 or (experts == 64 and hidden == 7168 and intermediate == 2048)", mode="eval").body
        return self.generic_visit(n)


scope = EmptyScope()
ns = {"ast": ast, "copy": copy, "types": types, "module": module, "TREE": TREE, "FUNCTIONS": FUNCTIONS}
exec(compile(ast.fix_missing_locations(scope.visit(module(defs))), "<host-oracle>", "exec"), ns)
assert scope.count == 2
shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192),
          (32, 7168, 2048), (32, 4096, 2048), (32, 7168, 1024),
          (64, 7168, 2048), (64, 4096, 2048), (64, 7168, 1024))
host_cases = 0
for (e, h, inter), valid, padded, blocks, dtype in itertools.product(shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    wanted = ns["host_mock"](748, e, h, inter, valid, padded, blocks, dtype)
    if (e, h, inter) == (32, 7168, 2048) and valid > 0 and padded > 0 and blocks > 0:
        assert wanted[0][0] == BASE
        wanted[0] = (NEW, wanted[0][1])
    assert ns["host_mock"](755, e, h, inter, valid, padded, blocks, dtype) == wanted
    host_cases += 1
print(f"Host {host_cases} combinations x2 fresh inputs: current inputs, exact selector, E64/E16/Stage2/empty paths PASS", flush=True)

# Reuse only shape/tag buffer classes from the earlier independent Stage1
# audit. Extend its two-way oracle to the explicitly chosen three-way M.
helper = ast.parse((D / "bench_records/v749/audit_v749_cpu.py").read_text(encoding="utf-8"))
classes = [copy.deepcopy(n) for n in helper.body if isinstance(n, ast.ClassDef) and n.name in {"View", "Buffer", "Lang"}]


class ThreeWayOracle(ast.NodeTransformer):
    comparisons = 0
    geometries = 0

    def visit_Compare(self, n):
        if (isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Attribute)
                and n.left.func.attr == "startswith" and ast.unparse(n.comparators[0]) == "active_rows <= 64"):
            obj = ast.unparse(n.left.func.value)
            self.comparisons += 1
            return ast.parse(f'{obj} == selected_prefix + ("gate_local" if "gate" in {obj} else "up_local")', mode="eval").body
        return self.generic_visit(n)

    def visit_IfExp(self, n):
        if ast.unparse(n) == "64 if active_rows <= 64 else 128":
            self.geometries += 1
            return ast.Name(id="logical_m", ctx=ast.Load())
        return self.generic_visit(n)


oracle = ThreeWayOracle()
exec(compile(ast.fix_missing_locations(oracle.visit(module(classes))), "<symbolic-buffers>", "exec"), globals())
assert oracle.comparisons == 3 and oracle.geometries == 1
Buffer, Lang = globals()["Buffer"], globals()["Lang"]
program = compile(ast.fix_missing_locations(module([copy.deepcopy(branch)])), "<actual-v755-branch>", "exec")
count = 0
for active_rows, steps_now in itertools.product(range(129), (1, 2, 112)):
    logical_m = 128 if active_rows > 64 else 64 if active_rows > 32 else 32
    selected_prefix = "" if logical_m == 128 else "tail_" if logical_m == 64 else "tiny_"
    T = Lang()
    env = {"T": T, "actual_rows": active_rows, "tail_m": 64, "tiny_m": 32, "bt1": 128,
           "bh1": 64, "be1": 128, "k_steps": steps_now, "gu_k_pack": 2,
           "scale": 1.44269504, "block_start": 256, "expert_id": 3, "by": 2}
    shapes = {"stacked_expert_tokens": (512, steps_now * 64), "gate_w": (32, 2048, steps_now * 64),
              "up_w": (32, 2048, steps_now * 64), "input_shared": (128, 64), "weight_shared": (128, 64),
              "up_prefetch": (128, 64), "up_logits": (512, 2048)}
    shapes.update({prefix + kind: (m, 128) for prefix, m in (("", 128), ("tail_", 64), ("tiny_", 32))
                   for kind in ("gate_local", "up_local")})
    env.update({name: Buffer(name, shape) for name, shape in shapes.items()})
    exec(program, {"__builtins__": {"range": range}}, env)
    assert T.copies == [item for k in range(steps_now if active_rows else 0) for item in (
        ("gate_w", k, "weight_shared", (128, 64)),
        ("stacked_expert_tokens", k, "input_shared", (logical_m, 64)),
        ("up_w", k, "up_prefetch", (128, 64)),
        ("up_w", k, "weight_shared", (128, 64)))]
    assert T.mmas == [(selected_prefix + kind, k) for k in range(steps_now if active_rows else 0)
                      for kind in ("gate_local", "up_local")]
    assert T.syncs == (2 * steps_now - 1 if active_rows else 0)
    out = env["up_logits"].writes
    assert set(out) == {(256 + i, 256 + j) for i in range(active_rows) for j in range(128)}
    for (i, j), value in out.items():
        gate, up = -0.25 + (i - 256) / 512, 0.25 + (j - 256) / 256
        assert value == up * (gate * (1.0 / (1.0 + 2 ** (-gate * 1.44269504))))
    count += 1
for m in (32, 64, 128):
    footprint = {r * 64 + ((c // 4) ^ (r % 16)) * 4 + c % 4 for r, c in itertools.product(range(m), range(64))}
    assert footprint == set(range(m * 64))
print(f"Actual branch AST {count} rows0..128/K1,2,112 cases: GIU copy bounds/widths, Gate-Up/terminal order, WAR barriers, valid-only SwiGLU PASS", flush=True)
for version, p in PATHS.items():
    compile(SOURCE[version], str(p), "exec")
    print(version, hashlib.sha256(p.read_bytes()).hexdigest())
print("LIMIT: symbolic math is not GPU rounding; M32 T.gemm view/layout, auto RAW barriers and physical resources require compiler/GPU checks.")
