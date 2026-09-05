"""Read-only source/dataflow/host audit; no TileLang or GPU runtime needed.

Checks source scheduling and current-input forwarding, not device numerical
correctness, compiler barrier insertion, physical register usage or performance.
"""

import ast
from collections import Counter
import copy
import hashlib
from pathlib import Path
import sys
import types


DATA = Path(__file__).resolve().parents[2]
FILES = {
    720: DATA / "probe_v720_v719_e16_stage2_bfrag_only.py",
    724: DATA / "probe_v724_v720_e32_stage1_a_fragment_reuse.py",
    727: DATA / "probe_v727_v720_e32_stage1_gate_b4_interleave.py",
    728: DATA / "probe_v728_v720_e32_stage1_gate_tail_defer.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}
OLD = "_moe_stage1_prefetch_giu_merge"
DONOR = OLD + "_a_reuse"
NEW = {727: OLD + "_gate_b4_interleave", 728: OLD + "_gate_tail_defer"}


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def call_name(statement):
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return ast.unparse(statement.value.func)
    return None


class RemoveOperandSchedule(ast.NodeTransformer):
    """A strict whitelist: every remaining AST node must match the v724 donor."""

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and "matrix" in node.targets[0].id):
            assert ast.unparse(node.value.func) == "T.alloc_fragment"
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if call_name(node) in {
            "mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b", "mma_emitter.mma"
        }:
            return None
        return self.generic_visit(node)


def expected_schedule(version, terminal):
    plan = [("copy", "gate_w"), ("copy", "stacked_expert_tokens"), ("copy", "up_w")]
    if version == 727:
        plan += [("b", f"gate_matrix{i}", i) for i in range(4)]
    else:
        for i in range(3):
            plan += [("a", "input_matrix", i), ("b", "weight_matrix", i),
                     ("mma", "input_matrix", "weight_matrix", "gate_local")]
        plan += [("b", "gate_tail_matrix", 3)]
    plan += [("barrier",), ("copy", "up_prefetch")]
    if version == 727:
        for i in range(4):
            plan += [("a", "input_matrix", i), ("b", "weight_matrix", i),
                     ("mma", "input_matrix", f"gate_matrix{i}", "gate_local"),
                     ("mma", "input_matrix", "weight_matrix", "up_local")]
    else:
        for i in range(3):
            plan += [("a", "input_matrix", i), ("b", "weight_matrix", i),
                     ("mma", "input_matrix", "weight_matrix", "up_local")]
        plan += [("a", "input_matrix", 3), ("b", "weight_matrix", 3),
                 ("mma", "input_matrix", "gate_tail_matrix", "gate_local"),
                 ("mma", "input_matrix", "weight_matrix", "up_local")]
    if not terminal:
        plan += [("barrier",)]
    return plan


def body_schedule(body):
    plan = []
    for node in body:
        name = call_name(node)
        if name is None:
            assert isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "terminal_k"
            continue
        call = node.value
        args = [ast.unparse(arg) for arg in call.args]
        if name == "T.copy":
            source = call.args[0]
            plan.append(("copy", source.id if isinstance(source, ast.Name) else source.value.id))
        elif name == "T.sync_threads":
            plan.append(("barrier",))
        elif name in {"mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b"}:
            kind = "a" if name.endswith("_a") else "b"
            assert args[1] == ("input_shared" if kind == "a" else "weight_shared")
            plan.append((kind, args[0], ast.literal_eval(call.args[2])))
        else:
            assert name == "mma_emitter.mma"
            plan.append(("mma", *args))
    return plan


BODIES = {}
for version in (727, 728):
    new = NEW[version]
    assert set(FUNCS[version]) == set(FUNCS[720]) | {new}
    for name in FUNCS[720]:
        if name != "_get_stage1":
            assert segment(version, name) == segment(720, name), name
    dispatch = copy.deepcopy(FUNCS[720]["_get_stage1"])
    branches = [node for node in ast.walk(dispatch)
                if isinstance(node, ast.IfExp) and ast.unparse(node.test) == "num_experts == 32"]
    assert len(branches) == 1 and branches[0].body.id == OLD
    branches[0].body.id = new
    assert ast.dump(dispatch) == ast.dump(FUNCS[version]["_get_stage1"])
    restored = SOURCES[version].replace(segment(version, new) + "\n\n\n", "")
    restored = restored.replace(segment(version, "_get_stage1"), segment(720, "_get_stage1"))
    assert restored[restored.index("import torch"):].rstrip() == SOURCES[720][SOURCES[720].index("import torch"):].rstrip()

    actual = RemoveOperandSchedule().visit(copy.deepcopy(FUNCS[version][new]))
    expected = RemoveOperandSchedule().visit(copy.deepcopy(FUNCS[724][DONOR]))
    actual.name = expected.name
    assert ast.dump(actual) == ast.dump(expected)
    allocations = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(FUNCS[version][new])
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and "matrix" in node.targets[0].id
    }
    expected_allocations = {
        "input_matrix": "T.alloc_fragment((a_local_size,), dtype=dtype)",
        "weight_matrix": "T.alloc_fragment((b_local_size,), dtype=dtype)",
    }
    retained = [f"gate_matrix{i}" for i in range(4)] if version == 727 else ["gate_tail_matrix"]
    expected_allocations.update({name: "T.alloc_fragment((b_local_size,), dtype=dtype)" for name in retained})
    assert allocations == expected_allocations
    guard = next(node for node in ast.walk(FUNCS[version][new])
                 if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")
    assert isinstance(guard.body[0], ast.For)
    assert ast.unparse(guard.body[0].iter) == "range(k_steps - 1)"
    BODIES[version] = (guard.body[0].body, guard.body[1:])
    for terminal, body in enumerate(BODIES[version]):
        assert body_schedule(body) == expected_schedule(version, terminal)
    print(f"v{version}: source/AST whitelist and exact steady/terminal operand schedule PASS")


def evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<index-audit>", "eval"), {"__builtins__": {}}, environment)


def audit_dataflow(version, k_steps):
    # Tags survive shared overwrites only when actually loaded into a distinct
    # fragment.  Keeping them across K also detects stale fragment reuse.
    shared, registers, outstanding_reads = {}, {}, set()
    products = {"gate_local": [], "up_local": []}
    counts = Counter()
    sequence = list(range(k_steps - 1)) + [k_steps - 1]
    assert sequence == list(range(k_steps))
    for position, k in enumerate(sequence):
        terminal = position == k_steps - 1
        body = BODIES[version][terminal]
        env = {"k": k, "terminal_k": k, "k_steps": k_steps, "bt1": 128,
               "be1": 128, "bh1": 64, "expert_id": 7, "by": 3, "block_start": 256}
        for node in body:
            name = call_name(node)
            if name is None:
                assert evaluate(node.value, env) == k
                continue
            call = node.value
            args = [ast.unparse(arg) for arg in call.args]
            if name == "T.copy":
                destination = args[1]
                assert destination not in outstanding_reads, (version, k, destination)
                source = call.args[0]
                if isinstance(source, ast.Name):
                    assert source.id == "up_prefetch" and destination == "weight_shared"
                    assert registers[source.id] == ("up_w", k)
                    shared[destination] = registers[source.id]
                else:
                    assert isinstance(source, ast.Subscript)
                    kind = source.value.id
                    coordinates = source.slice.elts
                    k_slice = coordinates[-1]
                    assert isinstance(k_slice, ast.Slice) and k_slice.step is None
                    assert (evaluate(k_slice.lower, env), evaluate(k_slice.upper, env)) == (k * 64, (k + 1) * 64)
                    row_slice = coordinates[-2]
                    start, end = evaluate(row_slice.lower, env), evaluate(row_slice.upper, env)
                    assert end - start == 128
                    if kind in {"gate_w", "up_w"}:
                        assert len(coordinates) == 3 and evaluate(coordinates[0], env) == 7
                        assert start == 384
                    else:
                        assert kind == "stacked_expert_tokens" and len(coordinates) == 2 and start == 256
                    if destination == "up_prefetch":
                        assert kind == "up_w"
                        registers[destination] = (kind, k)
                    else:
                        shared[destination] = (kind, k)
                counts["copy"] += 1
            elif name == "T.sync_threads":
                outstanding_reads.clear()
                counts["barrier"] += 1
            elif name in {"mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b"}:
                kind = "a" if name.endswith("_a") else "b"
                micro = ast.literal_eval(call.args[2])
                assert 0 <= micro < 4 and args[1] in shared
                tag = shared[args[1]]
                assert tag[1] == k
                if kind == "a":
                    assert tag[0] == "stacked_expert_tokens"
                registers[args[0]] = (tag[0], tag[1], micro)
                outstanding_reads.add(args[1])
                counts[kind] += 1
            else:
                assert name == "mma_emitter.mma"
                a, b = registers[args[0]], registers[args[1]]
                expected_b = "gate_w" if args[2] == "gate_local" else "up_w"
                assert a[0] == "stacked_expert_tokens" and b[0] == expected_b
                assert a[1:] == b[1:] and a[1] == k
                products[args[2]].append(a[1:])
                counts["mma"] += 1
        if not terminal:
            assert not outstanding_reads  # original end-K barrier still covers next overwrite
    expected_products = [(k, micro) for k in range(k_steps) for micro in range(4)]
    assert products == {"gate_local": expected_products, "up_local": expected_products}
    assert counts == Counter(copy=4*k_steps, barrier=2*k_steps-1,
                             a=(4 if version == 727 else 7)*k_steps, b=8*k_steps, mma=8*k_steps)


for version in (727, 728):
    for k_steps in (1, 2, 3, 4, 31, 32, 111, 112, 113):
        audit_dataflow(version, k_steps)
    print(f"v{version}: tagged shared/fragment dependencies, exact per-C K order, overwrite barriers PASS")

# Identical emitter configuration and output layout are covered by the whitelist.
# Check logical A/B microtile coverage and local slot bounds independently.  This
# counts logical fragments, never physical registers or measured occupancy.
for operand in ("a", "b"):
    coverage = Counter()
    for thread in range(256):
        lane = thread % 64
        warp_axis = (thread // 64) % 2 if operand == "a" else thread // 128
        for micro in range(4):
            for tile in range(4):
                for local in range(4):
                    row = warp_axis * 64 + tile * 16 + lane % 16
                    col = micro * 16 + (lane // 16) * 4 + local
                    assert 0 <= tile * 4 + local < 16
                    coverage[row, col] += 1
    assert set(coverage) == {(row, col) for row in range(128) for col in range(64)}
    assert set(coverage.values()) == {2}
assert (128 * 64 + 128 * 64) * 2 == 32768
print("Logical A/B 128x64 coverage, local 16-half slot bounds, shared32KiB PASS")


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "mock:0"


def run_mock(version, experts, route_dtype):
    compiled, launches, workspaces = [], [], []
    torch = types.ModuleType("torch")
    torch.float16, torch.float32 = "float16", "float32"
    def empty(shape, device, dtype):
        assert device == "mock:0"
        tensor = Tensor(shape, dtype)
        workspaces.append(tensor)
        return tensor
    torch.empty = empty
    language = types.ModuleType("tilelang.language")
    language.float16, language.float32 = "float16", "float32"
    tilelang = types.ModuleType("tilelang")
    tilelang.PassConfigKey = types.SimpleNamespace(TL_DISABLE_WARP_SPECIALIZED="disable_ws")
    def jit(*, pass_configs):
        def decorate(original):
            def factory(*args):
                compiled.append((original.__name__, args, dict(pass_configs)))
                def launch(*inputs):
                    launches.append((original.__name__, inputs))
                return launch
            return factory
        return decorate
    tilelang.jit = jit
    intrinsics = types.ModuleType("tilelang.maca.intrinsics")
    intrinsics.TensorCoreIntrinEmitter = object
    intrinsics.make_mma_swizzle_layout = object
    replacements = {"torch": torch, "tilelang": tilelang, "tilelang.language": language,
                    "tilelang.maca.intrinsics": intrinsics}
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        namespace = {}
        exec(compile(TREES[version], str(FILES[version]), "exec"), namespace)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    hidden = 2048 if experts == 16 else 7168 if experts in (32, 64) else 512
    intermediate = 8192 if experts == 16 else 2048 if experts in (32, 64) else 256
    padded, valid = experts * 256, experts * 142
    all_inputs = []
    for repeat in range(2):
        specs = [((padded, hidden), "float16"), ((experts, intermediate, hidden), "float16"),
                 ((experts, intermediate, hidden), "float16"), ((experts, hidden, intermediate), "float16"),
                 ((valid,), route_dtype), ((experts,), "int32"), ((experts + 1,), "int32"),
                 ((experts + 1,), "int32"), ((padded // 128,), "int32"), ((padded, hidden), "float16")]
        inputs = tuple(Tensor(shape, dtype) for shape, dtype in specs)
        all_inputs.append(inputs)
        namespace["run_kernel"](*inputs)
        assert len(launches) == 2 * (repeat + 1) and len(compiled) == 2 and len(workspaces) == 1
        assert launches[-2][1] == tuple(inputs[index] for index in (0, 1, 2, 5, 7, 8)) + (workspaces[0],)
        assert launches[-1][1] == (workspaces[0],) + tuple(inputs[index] for index in (3, 4, 5, 6, 7, 8, 9))
        assert compiled[1][1][-1] == route_dtype
    assert all(first is not second for first, second in zip(*all_inputs))
    return compiled



for version in (727, 728):
    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            actual, expected = run_mock(version, experts, dtype), run_mock(720, experts, dtype)
            if experts == 32:
                _, args, config = expected[0]
                expected[0] = (NEW[version], args, config)
            assert actual == expected, (version, experts, dtype)
            print(f"v{version} E{experts:02} {dtype}: two fresh calls, two launches each, exact forwarding PASS")
    print(f"v{version} SHA256", hashlib.sha256(FILES[version].read_bytes()).hexdigest())
print("No GPU result implied. Review inferred producer/read barriers and fragment aliasing in generated code.")
