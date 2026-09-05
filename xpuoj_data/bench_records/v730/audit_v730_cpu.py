"""CPU-only v730 audit: exact single barrier move from frozen v727.

No GPU compilation, synchronization inference, performance or numerical result
is implied.  Run from any directory with Python; no TileLang import is needed.
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
    727: DATA / "probe_v727_v720_e32_stage1_gate_b4_interleave.py",
    730: DATA / "probe_v730_v727_e32_stage1_early_tail_barrier.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}
BUILDER = "_moe_stage1_prefetch_giu_merge_gate_b4_interleave"


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def call_name(statement):
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return ast.unparse(statement.value.func)
    return None


def guard_for(builder):
    return next(node for node in ast.walk(builder)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")


assert set(FUNCS[730]) == set(FUNCS[727])
for name in FUNCS[727]:
    if name != BUILDER:
        assert segment(730, name) == segment(727, name), name

expected_tree = copy.deepcopy(TREES[727])
expected_builder = next(node for node in expected_tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == BUILDER)
expected_guard = guard_for(expected_builder)
assert isinstance(expected_guard.body[0], ast.For)
steady = expected_guard.body[0].body
old_tail = [
    "mma_emitter.ldmatrix_a(input_matrix, input_shared, 3)",
    "mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)",
    "mma_emitter.mma(input_matrix, gate_matrix3, gate_local)",
    "mma_emitter.mma(input_matrix, weight_matrix, up_local)",
    "T.sync_threads()",
]
assert [ast.unparse(node) for node in steady[-5:]] == old_tail
barrier = steady.pop()
steady.insert(len(steady) - 2, barrier)
assert ast.dump(expected_tree) == ast.dump(TREES[730])
indent = " " * 20
old_text = "\n".join(indent + line for line in old_tail)
new_tail = old_tail[:2] + old_tail[-1:] + old_tail[2:4]
new_text = "\n".join(indent + line for line in new_tail)
base_code = SOURCES[727][SOURCES[727].index("import torch"):].rstrip()
assert base_code.count(old_text) == 1
assert base_code.replace(old_text, new_text) == SOURCES[730][SOURCES[730].index("import torch"):].rstrip()
print("Entire source/AST: only header plus one existing steady tail barrier move PASS")
print("All builders, E16/E32/E64 dispatch paths, math, passes, geometry and copies otherwise unchanged PASS")

base_guard = guard_for(FUNCS[727][BUILDER])
actual_guard = guard_for(FUNCS[730][BUILDER])
assert ast.dump(ast.Module(body=base_guard.body[1:], type_ignores=[])) == ast.dump(
    ast.Module(body=actual_guard.body[1:], type_ignores=[]))
actual_steady = actual_guard.body[0].body
assert [ast.unparse(node) for node in actual_steady[-5:]] == new_tail
assert all(call_name(node) == "mma_emitter.mma" for node in actual_steady[-2:])
assert [tuple(ast.unparse(arg) for arg in node.value.args) for node in actual_steady[-2:]] == [
    ("input_matrix", "gate_matrix3", "gate_local"),
    ("input_matrix", "weight_matrix", "up_local"),
]
assert sum(call_name(node) == "T.sync_threads" for node in actual_steady) == 2
assert sum(call_name(node) == "T.sync_threads" for node in actual_guard.body[1:]) == 1
BODIES = {730: (actual_steady, actual_guard.body[1:])}
print("Terminal source/AST unchanged; only two register-only MMAs after the moved barrier PASS")


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
                             a=4*k_steps, b=8*k_steps, mma=8*k_steps)

for k_steps in (1, 2, 3, 4, 31, 32, 111, 112, 113):
    audit_dataflow(730, k_steps)
print("Tagged dataflow: correct retained GateB, UpB and A for every K16; no shared read after moved barrier PASS")
print("Every K: 4 copies, 4 A loads, 8 B loads, 8 MMA; steady2/terminal1 barriers PASS")

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



for experts in (1, 8, 16, 32, 64):
    for dtype in ("float16", "float32"):
        assert run_mock(730, experts, dtype) == run_mock(727, experts, dtype)
        print(f"E{experts:02} {dtype}: two fresh calls, two launches each, unchanged current-input forwarding PASS")
print("v730 SHA256", hashlib.sha256(FILES[730].read_bytes()).hexdigest())
print("Remaining GPU checks: generated barriers/fragment aliasing, correctness, paired entry performance.")
