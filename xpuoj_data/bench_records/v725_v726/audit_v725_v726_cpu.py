"""Self-contained static/CPU audit; does not replace GPU code-generation and correctness tests."""

import ast
from collections import Counter
import copy
import hashlib
from pathlib import Path
import sys
import types


DATA = Path(__file__).resolve().parents[2]
SPECS = {
    725: ("gate", "gate_w", "weight_shared"),
    726: ("input", "stacked_expert_tokens", "input_shared"),
}
FILES = {720: DATA / "probe_v720_v719_e16_stage2_bfrag_only.py"}
FILES.update({version: DATA / f"probe_v{version}_v720_e32_stage1_{kind}_staging_cw8.py"
              for version, (kind, _, _) in SPECS.items()})
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}
OLD = "_moe_stage1_prefetch_giu_merge"


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


class ExpectedStaging(ast.NodeTransformer):
    def __init__(self, source, destination):
        self.source, self.destination = source, destination
        self.replaced = 0

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        call = node.value
        if ast.unparse(call.func) != "T.copy" or not isinstance(call.args[0], ast.Subscript):
            return self.generic_visit(node)
        if ast.unparse(call.args[0].value) != self.source:
            return self.generic_visit(node)
        assert ast.unparse(call.args[1]) == self.destination
        first = copy.deepcopy(node)
        first.value.args[1] = ast.Name(id="up_prefetch", ctx=ast.Load())
        first.value.keywords = [ast.keyword(arg="coalesced_width", value=ast.Constant(8))]
        second = ast.parse(f"T.copy(up_prefetch, {self.destination}, coalesced_width=4)").body[0]
        self.replaced += 1
        return [first, second]


def evaluate(node, **env):
    return eval(compile(ast.Expression(node), "<audit>", "eval"), {}, env)


def trace_body(body, tile, state):
    """Interpret only copies/GEMMs as tagged tiles, detecting stale fragment contents."""
    env = dict(k=tile, terminal_k=tile, bh1=64, bt1=128, be1=128, by=2, expert_id=3,
               block_start=512)
    phases = []
    barriers = 0
    for statement in body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        function = ast.unparse(call.func)
        if function == "T.copy":
            source = call.args[0]
            destination = ast.unparse(call.args[1])
            if isinstance(source, ast.Subscript):
                name = ast.unparse(source.value)
                slices = source.slice.elts
                assert evaluate(slices[-1].lower, **env) == tile * 64
                assert evaluate(slices[-1].upper, **env) == (tile + 1) * 64
                assert evaluate(slices[-2].upper, **env) - evaluate(slices[-2].lower, **env) == 128
                value = (name, tile)
            else:
                # Any read before definition is an error. Contents persist between Ks,
                # so a missed overwrite would be detected as a stale tile below.
                value = state[ast.unparse(source)]
            state[destination] = value
        elif function == "T.gemm":
            arguments = [ast.unparse(arg) for arg in call.args]
            assert arguments[:2] == ["input_shared", "weight_shared"]
            assert state["input_shared"] == ("stacked_expert_tokens", tile)
            expected_weight = "gate_w" if arguments[2] == "gate_local" else "up_w"
            assert state["weight_shared"] == (expected_weight, tile)
            assert state["up_prefetch"] == ("up_w", tile)
            phases.append(arguments[2])
        elif function == "T.sync_threads":
            barriers += 1
    assert phases == ["gate_local", "up_local"]
    return barriers


for version, (kind, source, destination) in SPECS.items():
    new_name = OLD + "_" + kind + "_stage_cw8"
    assert set(FUNCS[version]) == set(FUNCS[720]) | {new_name}
    for name in FUNCS[720]:
        if name != "_get_stage1":
            assert segment(version, name) == segment(720, name), (version, name)
    transformer = ExpectedStaging(source, destination)
    expected = transformer.visit(copy.deepcopy(FUNCS[720][OLD]))
    expected.name = new_name
    assert transformer.replaced == 2
    assert ast.dump(expected) == ast.dump(FUNCS[version][new_name])

    dispatch = copy.deepcopy(FUNCS[720]["_get_stage1"])
    branches = [node for node in ast.walk(dispatch)
                if isinstance(node, ast.IfExp) and ast.unparse(node.test) == "num_experts == 32"]
    assert len(branches) == 1 and branches[0].body.id == OLD
    branches[0].body.id = new_name
    assert ast.dump(dispatch) == ast.dump(FUNCS[version]["_get_stage1"])
    restored = SOURCES[version].replace(segment(version, new_name) + "\n\n\n", "")
    restored = restored.replace(segment(version, "_get_stage1"), segment(720, "_get_stage1"))
    assert restored[restored.index("import torch"):] == SOURCES[720][SOURCES[720].index("import torch"):]
    print(f"v{version}: complete-source + AST proof, only the two selected copies and E32 dispatcher change PASS")

    # The unchanged dispatcher always selects matching 128x64 shapes for E32.
    tile_shape = ast.literal_eval(FUNCS[version]["_pick_tiles"].body[-1].value)
    bt, bk, bn, threads = tile_shape
    assert tile_shape == (128, 64, 128, 256) and bt == bn
    allocations = [(ast.unparse(node.targets[0]), ast.unparse(node.value))
                   for node in ast.walk(FUNCS[version][new_name])
                   if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                   and ast.unparse(node.value.func) in ("T.alloc_fragment", "T.alloc_shared")]
    assert len(allocations) == 5  # original A/B shared, one prefetch, Gate/Up accumulators
    assert ("up_prefetch", "T.alloc_fragment((be1, bh1), dtype=dtype)") in allocations
    assert (bt * bk + bn * bk) * 2 == 32768

    guard = next(node for node in ast.walk(FUNCS[version][new_name])
                 if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")
    steady, terminal = guard.body[0].body, guard.body[1:]
    assert ast.unparse(guard.body[0].iter) == "range(k_steps - 1)"
    for count in (1, 2, 3, 4, 31, 32, 111, 112, 113):
        state = {}
        for tile in range(count - 1):
            assert trace_body(steady, tile, state) == 2
        assert trace_body(terminal, count - 1, state) == 1
    print(f"v{version}: steady/terminal K tags, immediate fragment consumption and Up overwrite-before-use PASS")

# Model the already observed Up cw8 global -> same fragment -> cw4 shared mapping.
# This proves shape/index compatibility; actual generated vector types are a separate check.
loads, stores = Counter(), Counter()
for thread in range(256):
    fragments = {}
    for iteration in range(4):
        for lane in range(8):
            row = iteration * 32 + thread // 8
            col = (thread % 8) * 8 + lane
            fragments[iteration * 8 + lane] = (row, col)
            loads[row, col] += 1
    for iteration in range(8):
        for lane in range(4):
            row = (iteration // 2) * 32 + thread // 8
            col = (thread % 8) * 8 + (iteration % 2) * 4 + lane
            assert fragments[iteration * 4 + lane] == (row, col)
            stores[row, col] += 1
assert loads == stores
assert set(loads) == {(row, col) for row in range(128) for col in range(64)}
assert set(loads.values()) == {1}
print("cw8 fragment -> cw4 shared lane mapping: complete 128x64 tile, same 32-half fragment/thread PASS")


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



for version, (kind, _, _) in SPECS.items():
    new_name = OLD + "_" + kind + "_stage_cw8"
    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            actual, expected = run_mock(version, experts, dtype), run_mock(720, experts, dtype)
            if experts == 32:
                _, args, config = expected[0]
                expected[0] = (new_name, args, config)
            assert actual == expected, (version, experts, dtype)
            print(f"v{version} E{experts:02} {dtype}: two fresh inputs, two launches each, exact forwarding PASS")
    print(f"v{version} SHA256 {hashlib.sha256(FILES[version].read_bytes()).hexdigest()}")
print("No GPU result implied. Next: inspect vector widths, register/spills, barriers, then random correctness/entry timing.")
