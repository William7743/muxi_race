"""CPU/source audit only; GPU compilation, synchronization and precision remain separate checks."""

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
    563: DATA / "probe_v563_s1_e32_single_bfrag_emitter.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}
OLD = "_moe_stage1_prefetch_giu_merge"
NEW = OLD + "_a_reuse"
REFERENCE = "_moe_stage1_prefetch_giu_merge_bfrag_prefetch"


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


emitter_configuration = [
    copy.deepcopy(node) for node in FUNCS[563][REFERENCE].body
    if assigned_name(node) in {"mma_emitter", "a_local_size", "b_local_size"}
]
assert len(emitter_configuration) == 3
reference_layout = next(node.value.args[0] for node in ast.walk(FUNCS[563][REFERENCE])
                        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == "T.annotate_layout")
store_layout_entries = [(copy.deepcopy(key), copy.deepcopy(value))
                       for key, value in zip(reference_layout.keys, reference_layout.values)
                       if isinstance(key, ast.Name) and key.id in {"gate_local", "up_local"}]
assert len(store_layout_entries) == 2


class ExpectedTransformation(ast.NodeTransformer):
    def __init__(self):
        self.gemm_count = 0
        self.allocation_count = 0
        self.layout_count = 0

    def visit_Assign(self, node):
        name = assigned_name(node)
        if name == "gu_k_pack":
            assert ast.literal_eval(node.value) == 2
            return None
        if name == "accum_dtype":
            return [node] + copy.deepcopy(emitter_configuration)
        if name == "up_prefetch":
            declarations = [f"input_matrix{index} = T.alloc_fragment((a_local_size,), dtype=dtype)"
                            for index in range(4)]
            declarations.append("weight_matrix = T.alloc_fragment((b_local_size,), dtype=dtype)")
            self.allocation_count += 1
            return [node] + ast.parse("\n".join(declarations)).body
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        call = node.value
        if ast.unparse(call.func) == "T.annotate_layout":
            for key, value in store_layout_entries:
                call.args[0].keys.append(copy.deepcopy(key))
                call.args[0].values.append(copy.deepcopy(value))
            self.layout_count += 1
            return node
        if ast.unparse(call.func) != "T.gemm":
            return self.generic_visit(node)
        self.gemm_count += 1
        assert [ast.unparse(arg) for arg in call.args[:2]] == ["input_shared", "weight_shared"]
        accumulator = ast.unparse(call.args[2])
        assert accumulator in {"gate_local", "up_local"}
        statements = []
        for micro in range(4):
            if accumulator == "gate_local":
                statements.append(f"mma_emitter.ldmatrix_a(input_matrix{micro}, input_shared, {micro})")
            statements.append(f"mma_emitter.ldmatrix_b(weight_matrix, weight_shared, {micro})")
            statements.append(f"mma_emitter.mma(input_matrix{micro}, weight_matrix, {accumulator})")
        return ast.parse("\n".join(statements)).body


assert set(FUNCS[724]) == set(FUNCS[720]) | {NEW}
for function in FUNCS[720]:
    if function != "_get_stage1":
        assert segment(724, function) == segment(720, function), function
transformer = ExpectedTransformation()
expected = transformer.visit(copy.deepcopy(FUNCS[720][OLD]))
expected.name = NEW
assert transformer.gemm_count == 4 and transformer.allocation_count == transformer.layout_count == 1
assert ast.dump(expected) == ast.dump(FUNCS[724][NEW])
dispatch = copy.deepcopy(FUNCS[720]["_get_stage1"])
branches = [node for node in ast.walk(dispatch)
            if isinstance(node, ast.IfExp) and ast.unparse(node.test) == "num_experts == 32"]
assert len(branches) == 1 and branches[0].body.id == OLD
branches[0].body.id = NEW
assert ast.dump(dispatch) == ast.dump(FUNCS[724]["_get_stage1"])
restored = SOURCES[724].replace(segment(724, NEW) + "\n\n\n", "")
restored = restored.replace(segment(724, "_get_stage1"), segment(720, "_get_stage1"))
assert restored[restored.index("import torch"):] == SOURCES[720][SOURCES[720].index("import torch"):]
print("Whole-source and AST whitelist: only added E32 A-reuse builder + dispatcher + header PASS")
print("GIU copies, shared allocations, passes, serial K loop, empty-row guard and output epilogue unchanged PASS")

# Verify per-microtile data dependencies in both steady and terminal bodies.
builder = FUNCS[724][NEW]
guard = next(node for node in ast.walk(builder)
             if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")
assert isinstance(guard.body[0], ast.For)
steady = guard.body[0].body
terminal = guard.body[1:]
for label, body, expected_barriers in (("steady", steady, 2), ("terminal", terminal, 1)):
    a_slots = {}
    b_micro = None
    phase = None
    mma_order = {"gate_local": [], "up_local": []}
    counters = Counter()
    for statement in body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        function = ast.unparse(call.func)
        args = [ast.unparse(arg) for arg in call.args]
        if function == "T.copy" and args[1] == "weight_shared":
            phase = "up_local" if args[0] == "up_prefetch" else "gate_local"
        elif function == "mma_emitter.ldmatrix_a":
            assert phase == "gate_local" and args[1] == "input_shared"
            assert args[0] not in a_slots
            a_slots[args[0]] = int(args[2])
            counters["a_load"] += 1
        elif function == "mma_emitter.ldmatrix_b":
            assert args[:2] == ["weight_matrix", "weight_shared"]
            b_micro = int(args[2])
            counters["b_load"] += 1
        elif function == "mma_emitter.mma":
            assert args[1] == "weight_matrix" and args[2] == phase
            assert a_slots[args[0]] == b_micro
            mma_order[phase].append(b_micro)
        elif function == "T.sync_threads":
            counters["barrier"] += 1
    assert mma_order == {"gate_local": [0, 1, 2, 3], "up_local": [0, 1, 2, 3]}
    assert counters == Counter(a_load=4, b_load=8, barrier=expected_barriers)
    print(f"{label}: four A definitions each reused by Gate/Up; K16 order 0,1,2,3 and original explicit barriers PASS")

# Check the 2x2-warp, 64x64/warp, K16/k_pack1 layout model used by v563.
# This is a logical-index audit, not proof of runtime layout lowering or barrier insertion.
configuration = next(node.value for node in emitter_configuration if assigned_name(node) == "mma_emitter")
keywords = {item.arg: ast.unparse(item.value) for item in configuration.keywords}
for key, value in {"block_row_warps": "2", "block_col_warps": "2", "warp_row_tiles": "64",
                   "warp_col_tiles": "64", "chunk": "bh1", "k_pack": "1",
                   "a_transposed": "False", "b_transposed": "True"}.items():
    assert keywords[key] == value
micro_map = {(lane % 16, (lane // 16) * 4 + local) for lane in range(64) for local in range(4)}
assert micro_map == {(row, col) for row in range(16) for col in range(16)}
a_coverage = Counter()
for thread in range(256):
    lane = thread % 64
    warp_m = (thread // 64) % 2
    for micro in range(4):
        for tile in range(4):
            for local in range(4):
                row = warp_m * 64 + tile * 16 + lane % 16
                col = micro * 16 + (lane // 16) * 4 + local
                a_coverage[row, col] += 1
                assert 0 <= tile * 4 + local < 16
assert set(a_coverage) == {(row, col) for row in range(128) for col in range(64)}
assert set(a_coverage.values()) == {2}  # same A used by both N warps, as in v563
assert 4 * 16 * 2 == 128  # logical bytes of four A fragment arrays per thread
assert (128 * 64 + 128 * 64) * 2 == 32768
print("v563 emitter/store-layout AST unchanged; logical 128x64 A coverage and 16-half slot bounds PASS")


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
        actual, expected = run_mock(724, experts, dtype), run_mock(720, experts, dtype)
        if experts == 32:
            _, args, config = expected[0]
            expected[0] = (NEW, args, config)
        assert actual == expected, (experts, dtype)
        print(f"E{experts:02} {dtype}: two fresh calls, two launches each, exact current-input forwarding PASS")
print("Remaining GPU checks: generated write/read barriers; fragment aliasing; precision; register/spill pressure")
print("SHA256", hashlib.sha256(FILES[724].read_bytes()).hexdigest())
