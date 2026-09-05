"""CPU/source audit of v731/v732 lane-local Gate64+Up64 candidates.

Run: python xpuoj_data/bench_records/v731_v732/audit_v731_v732_cpu.py
No GPU/TileLang is required. Index and mock-dispatch proofs do NOT establish
device numerical correctness, inferred barriers, occupancy, or performance.
The official MACA 64-lane C map is row=lane%16, col=local+(lane//16)*4.
Attempt 1 used high-level paired C indexing and failed LayoutInference:
731 e0d12321f3c73d926647e58b0738689d1a565ef53eba6ed5098cde0fb499e292
732 b81a9c50dd84946ab37a86883c982069eb202a4a487c8dcd371b5c6c782ca52e
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
    731: DATA / "probe_v731_v720_e32_stage1_concat_gu64_k64.py",
    732: DATA / "probe_v732_v720_e32_stage1_concat_gu64_k32.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}
OLD = "_moe_stage1_prefetch_giu_merge"
NEW = "_moe_stage1_concat_gu_n128"
EXTRA_IMPORT = "from tilelang.maca.intrinsics.layout.utils import mma_store_index_map"


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


def assignments(function):
    return {node.targets[0].id: node.value for node in ast.walk(function)
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)}


def expression(text):
    return ast.parse(text, mode="eval").body


for version in (731, 732):
    assert set(FUNCS[version]) == set(FUNCS[720]) | {NEW}
    for name in FUNCS[720]:
        if name != "_get_stage1":
            assert segment(version, name) == segment(720, name), (version, name)
    dispatch = copy.deepcopy(FUNCS[720]["_get_stage1"])
    dispatch.body[1:1] = ast.parse(
        "use_concat = num_experts == 32 and hidden == 7168 and intermediate == 2048"
        + ("\nif use_concat:\n    bh1 = 32" if version == 732 else "")
    ).body
    branches = [node for node in ast.walk(dispatch)
                if isinstance(node, ast.IfExp) and ast.unparse(node.test) == "num_experts == 32"]
    assert len(branches) == 1 and branches[0].body.id == OLD
    branches[0].body = expression(f"{NEW} if use_concat else {OLD}")
    assert ast.dump(dispatch) == ast.dump(FUNCS[version]["_get_stage1"])
    restored = SOURCES[version].replace(segment(version, NEW) + "\n\n\n", "")
    restored = restored.replace(EXTRA_IMPORT + "\n", "")
    restored = restored.replace(segment(version, "_get_stage1"), segment(720, "_get_stage1"))
    assert restored[restored.index("import torch"):].rstrip() == SOURCES[720][SOURCES[720].index("import torch"):].rstrip()
    assert [ast.dump(item) for item in FUNCS[version][NEW].decorator_list] == [
        ast.dump(item) for item in FUNCS[720][OLD].decorator_list
    ]
    print(f"v{version}: exact original builders/host/Stage2 + E32-only dispatcher whitelist PASS")

assert segment(731, NEW) == segment(732, NEW)
builder = FUNCS[731][NEW]
values = assignments(builder)
base_values = assignments(FUNCS[720][OLD])
for name in ("scale", "dtype", "accum_dtype", "input_shape", "intermediate_shape",
             "gate_shape", "up_shape", "expert_id", "block_start", "group_size",
             "padded_start", "actual_rows", "k_steps", "terminal_k"):
    assert ast.dump(values[name]) == ast.dump(base_values[name]), name
assert ast.unparse(values["output_tile"]) == "be1 // 2"
assert ast.unparse(values["mma_emitter"].func) == "TensorCoreIntrinEmitter"
assert {kw.arg: ast.unparse(kw.value) for kw in values["mma_emitter"].keywords} == {
    "a_dtype": "dtype", "b_dtype": "dtype", "accum_dtype": "accum_dtype",
    "a_transposed": "False", "b_transposed": "True", "block_row_warps": "4",
    "block_col_warps": "1", "warp_row_tiles": "32", "warp_col_tiles": "128",
    "chunk": "bh1", "k_pack": "1",
}
expected_allocations = {
    "input_shared": "T.alloc_shared((bt1, bh1), dtype=dtype)",
    "weight_shared": "T.alloc_shared((be1, bh1), dtype=dtype)",
    "input_matrix": "T.alloc_fragment((a_local_size,), dtype=dtype)",
    "weight_matrix": "T.alloc_fragment((b_local_size,), dtype=dtype)",
    "gu_local": "T.alloc_local((64,), dtype=accum_dtype)",
}
allocations = {name: ast.unparse(value) for name, value in values.items()
               if isinstance(value, ast.Call) and ast.unparse(value.func).startswith("T.alloc_")}
assert allocations == expected_allocations
assert ast.unparse(values["a_local_size"]) == "mma_emitter.warp_rows * mma_emitter.k_pack * mma_emitter.local_size_a"
assert ast.unparse(values["b_local_size"]) == "mma_emitter.warp_cols * mma_emitter.k_pack * mma_emitter.local_size_b"
for call in ("T.annotate_layout", "T.use_swizzle"):
    actual = [node for node in ast.walk(builder) if call_name(node) == call]
    expected = [node for node in ast.walk(FUNCS[720][OLD]) if call_name(node) == call]
    assert len(actual) == len(expected) == 1
    assert ast.dump(actual[0]) == ast.dump(expected[0]), call
kernel = next(node for node in ast.walk(builder) if isinstance(node, ast.With))
assert ast.unparse(kernel.items[0].context_expr) == (
    "T.Kernel(num_blocks_m, T.ceildiv(intermediate, output_tile), threads=th1)"
)
stage = next(node for node in builder.body if isinstance(node, ast.FunctionDef))
base_stage = next(node for node in FUNCS[720][OLD].body if isinstance(node, ast.FunctionDef))
assert ast.dump(stage.args) == ast.dump(base_stage.args)
guard = next(node for node in ast.walk(builder)
             if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")
assert not guard.orelse
steady = guard.body[0]
assert isinstance(steady, ast.For) and ast.unparse(steady.iter) == "range(k_steps - 1)"
assert not steady.orelse
assert ast.unparse(guard.body[1]) == "terminal_k = k_steps - 1"
terminal = guard.body[2:]
assert [call_name(node) for node in steady.body] == [
    "T.copy", "T.copy", "T.copy", None, "T.sync_threads"
]
assert [call_name(node) for node in terminal] == ["T.copy", "T.copy", "T.copy", None]
assert ast.unparse(steady.body[3].iter) == "T.serial(bh1 // 16)"
assert ast.dump(steady.body[3]) == ast.dump(terminal[3])
assert [ast.unparse(node) for node in steady.body[3].body] == [
    "mma_emitter.ldmatrix_a(input_matrix, input_shared, ki)",
    "mma_emitter.ldmatrix_b(weight_matrix, weight_shared, ki)",
    "mma_emitter.mma(input_matrix, weight_matrix, gu_local)",
]
assert [ast.unparse(node) for node in ast.walk(builder) if call_name(node) == "T.clear"] == ["T.clear(gu_local)"]
assert sum(call_name(node) == "T.sync_threads" for node in ast.walk(builder)) == 1

# Prove every steady/terminal copy directly reads the current input, with
# correct Gate/Up N slice and disjoint, complete combined-B destination halves.
for body, k_name in ((steady.body, "k"), (terminal, "terminal_k")):
    expected = [
        f"T.copy(gate_w[expert_id, by * output_tile:(by + 1) * output_tile, {k_name} * bh1:({k_name} + 1) * bh1], weight_shared[0:output_tile, 0:bh1], coalesced_width=4)",
        f"T.copy(stacked_expert_tokens[block_start:block_start + bt1, {k_name} * bh1:({k_name} + 1) * bh1], input_shared)",
        f"T.copy(up_w[expert_id, by * output_tile:(by + 1) * output_tile, {k_name} * bh1:({k_name} + 1) * bh1], weight_shared[output_tile:be1, 0:bh1], coalesced_width=4)",
    ]
    for node, text in zip(body[:3], expected):
        assert ast.dump(node) == ast.dump(ast.parse(text).body[0])
print("Identical pass/layout/swizzle, fixed 4x1 emitter, current G/I/U copies and empty-row guard PASS")


def evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<index-audit>", "eval"),
                {"__builtins__": {}}, environment)


# C's flat offset is emitter.mma i*(warp_cols*4)+j*4+local.
# Its official C map uses 64 lanes, not the old 32-lane map.
def c_map(thread, row_tile, col_tile, local_id):
    lane, warp_m = thread % 64, thread // 64
    return (warp_m * 32 + row_tile * 16 + lane % 16,
            col_tile * 16 + local_id + (lane // 16) * 4)


coverage = Counter()
for thread in range(256):
    for row_tile in range(2):
        for col_tile in range(8):
            for local_id in range(4):
                slot = row_tile * 32 + col_tile * 4 + local_id
                assert 0 <= slot < 64
                coverage[c_map(thread, row_tile, col_tile, local_id)] += 1
assert coverage == Counter({(i, j): 1 for i in range(128) for j in range(128)})

# Evaluate output address and local-slot expressions directly from candidate AST.
for name, expected in {
    "thread_binding": "mma_emitter.get_thread_binding()",
    "output_row": "warp_m * 32 + row_tile * 16 + row",
    "output_col": "col_tile * 16 + col",
    "gate_slot": "row_tile * 32 + col_tile * 4 + local_id",
}.items():
    assert ast.unparse(values[name]) == expected
binding = next(node for node in ast.walk(builder) if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Tuple)
               and ast.unparse(node.targets[0]) == "(lane_id, warp_n, warp_m)")
assert ast.unparse(binding.value) == "mma_emitter.extract_thread_binding(thread_binding)"
map_assignment = next(node for node in ast.walk(builder) if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Tuple)
                      and ast.unparse(node.targets[0]) == "(row, col)")
assert ast.unparse(map_assignment.value) == "T.meta_var(mma_store_index_map(lane_id, local_id))"
epilogue = next(node for node in kernel.body
               if isinstance(node, ast.For) and ast.unparse(node.target) == "row_tile")
assert ast.unparse(epilogue.iter) == "T.serial(2)"
assert ast.unparse(epilogue.body[0].iter) == "T.serial(4)"
assert ast.unparse(epilogue.body[0].body[0].iter) == "T.serial(4)"
store = next(node for node in ast.walk(epilogue)
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript))
assert ast.unparse(store.targets[0]) == "up_logits[block_start + output_row, by * output_tile + output_col]"
assert ast.unparse(epilogue.body[0].body[0].body[-1].test) == "output_row < actual_rows"

outputs = Counter()
for thread in range(256):
    for row_tile in range(2):
        for col_tile in range(4):
            for local_id in range(4):
                lane, warp = thread % 64, thread // 64
                env = dict(warp_m=warp, row_tile=row_tile, col_tile=col_tile, local_id=local_id,
                           row=lane % 16, col=local_id + (lane // 16) * 4)
                row = evaluate(values["output_row"], env)
                col = evaluate(values["output_col"], env)
                slot = evaluate(values["gate_slot"], env)
                assert (row, col) == c_map(thread, row_tile, col_tile, local_id)
                assert (row, col + 64) == c_map(thread, row_tile, col_tile + 4, local_id)
                assert slot + 16 == row_tile * 32 + (col_tile + 4) * 4 + local_id
                assert slot in (*range(16), *range(32, 48)) and 0 <= slot + 16 < 64
                outputs[row, col] += 1
assert outputs == Counter({(i, j): 1 for i in range(128) for j in range(64)})
for actual_rows in (0, 1, 15, 64, 127, 128):
    assert sum(count for (row, _), count in outputs.items() if row < actual_rows) == actual_rows * 64
assert {by * 64 + col for by in range(32) for col in range(64)} == set(range(2048))
print("Official 64-lane C mapping: 16384 unique C, 8192 exact Gate/Up same-thread pairs; tail rows PASS")


class NormalizeMath(ast.NodeTransformer):
    def visit_Subscript(self, node):
        name = node.value.id
        if name in ("gate_local", "up_local"):
            return ast.Name(id="g" if name == "gate_local" else "u", ctx=ast.Load())
        assert name == "gu_local"
        if ast.unparse(node.slice) == "gate_slot":
            return ast.Name(id="g", ctx=ast.Load())
        assert ast.unparse(node.slice) == "gate_slot + 16"
        return ast.Name(id="u", ctx=ast.Load())


base_store = next(node for node in ast.walk(FUNCS[720][OLD])
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
                  and node.targets[0].value.id == "up_logits")
assert ast.dump(NormalizeMath().visit(copy.deepcopy(store.value))) == ast.dump(
    NormalizeMath().visit(copy.deepcopy(base_store.value))
)
for tile_k in (64, 32):
    steps = 7168 // tile_k
    sequence = list(range(steps - 1)) + [steps - 1]
    assert sequence == list(range(steps))
    k16 = [k * tile_k + micro * 16 for k in sequence for micro in range(tile_k // 16)]
    assert k16 == list(range(0, 7168, 16))
    for step in sequence:
        assert (step + 1) * tile_k - step * tile_k == tile_k
    for operand in ("a", "b"):
        count = Counter()
        for thread in range(256):
            lane, warp = thread % 64, thread // 64
            for micro in range(tile_k // 16):
                for tile in range(2 if operand == "a" else 8):
                    for local in range(4):
                        row = (warp * 32 if operand == "a" else 0) + tile * 16 + lane % 16
                        col = micro * 16 + (lane // 16) * 4 + local
                        count[row, col] += 1
        assert set(count) == {(i, j) for i in range(128) for j in range(tile_k)}
        assert set(count.values()) == ({1} if operand == "a" else {4})
    assert (128 + 128) * tile_k * 2 == (32768 if tile_k == 64 else 16384)
for steps in (1, 2, 3, 4, 31, 32, 111, 112, 224):
    assert list(range(steps - 1)) + [steps - 1] == list(range(steps))
print("SwiGLU operation AST unchanged; K64/K32 exact per-C K16 order, shared/operand coverage PASS")


class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.device = "mock:0"


def run_mock(version, experts, route_dtype, shape=None):
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
    layout_utils = types.ModuleType("tilelang.maca.intrinsics.layout.utils")
    layout_utils.mma_store_index_map = lambda lane, local: (lane % 16, local + (lane // 16) * 4)
    replacements = {"tilelang.maca.intrinsics.layout.utils": layout_utils, "torch": torch, "tilelang": tilelang, "tilelang.language": language,
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
    if shape is not None:
        hidden, intermediate = shape
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


for version in (731, 732):
    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            shapes = (None, (4096, 2048), (7168, 4096)) if experts == 32 else (None,)
            for shape in shapes:
                actual = run_mock(version, experts, dtype, shape)
                expected = run_mock(720, experts, dtype, shape)
                if experts == 32 and shape is None:
                    _, args, config = expected[0]
                    if version == 732:
                        args = args[:6] + (32,) + args[7:]
                    expected[0] = (NEW, args, config)
                assert actual == expected, (version, experts, dtype, shape)
            print(f"v{version} E{experts:02} {dtype}: two fresh calls / two launches, exact forwarding PASS")
    print(f"v{version} SHA256", hashlib.sha256(FILES[version].read_bytes()).hexdigest())
print("CPU audit PASS. GPU compilation, automatic producer/read barriers and numeric tests still required.")
