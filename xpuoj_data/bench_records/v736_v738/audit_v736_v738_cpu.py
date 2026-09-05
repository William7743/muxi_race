"""CPU-only audit of the independent v736/v737/v738 Stage2 probes.

No installed torch/TileLang, GPU compilation, GPU execution or performance claim.
Checks source semantics; generated synchronization and safe-pass-off load motion
must be checked separately. Run this file with Python from any working directory.
"""

import ast
from collections import Counter
import copy
import hashlib
import itertools
from pathlib import Path
import struct
import sys
import types


DATA = Path(__file__).resolve().parents[2]
FILES = {
    720: DATA / "probe_v720_v719_e16_stage2_bfrag_only.py",
    736: DATA / "probe_v736_v720_e32_stage2_early_barrier.py",
    737: DATA / "probe_v737_v720_e32_stage2_short_up_prefetch.py",
    738: DATA / "probe_v738_v720_e32_stage2_short_down_prefetch.py",
}
BUILDERS = {
    720: "_moe_stage2_fast_bfrag_prefetch",
    736: "_moe_stage2_fast_bfrag_tail_early_barrier",
    737: "_moe_stage2_fast_bfrag_tail_up_prefetch",
    738: "_moe_stage2_fast_bfrag_tail_down_prefetch",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def expression_name(statement):
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    return ast.unparse(statement.value.func)


def positive_guard(builder):
    return next(node for node in ast.walk(builder)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "active_k_steps > 0")


def steady_loop(builder):
    return next(node for node in positive_guard(builder).body if isinstance(node, ast.For))


def builder_selection(getter):
    return next(node for node in ast.walk(getter)
                if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "builder")


for version in (736, 737, 738):
    assert set(FUNCS[version]) == set(FUNCS[720]) | {BUILDERS[version]}
    for name in FUNCS[720]:
        if name != "_get_stage2":
            assert segment(version, name) == segment(720, name), (version, name)

    # Independently reconstruct the permitted transformation from v720.
    expected = copy.deepcopy(TREES[720])
    expected_functions = {node.name: node for node in expected.body if isinstance(node, ast.FunctionDef)}
    clone = copy.deepcopy(expected_functions[BUILDERS[720]])
    clone.name = BUILDERS[version]
    clone.body[0].value.value = {
        736: "E32 Stage2: steady-tail early_barrier, terminal path unchanged.",
        737: "E32 Stage2: steady-tail short_up_prefetch, terminal path unchanged.",
        738: "E32 Stage2: steady-tail short_down_prefetch, terminal path unchanged.",
    }[version]
    loop = steady_loop(clone)
    assert [expression_name(node) for node in loop.body[-4:]] == [
        "mma_emitter.mma", "T.sync_threads", "T.copy", "T.copy"]
    assert ast.unparse(loop.body[-5]) == "mma_emitter.ldmatrix_a(up_matrix, up_shared, 3)"
    final_mma, barrier, up_copy, down_copy = loop.body[-4:]
    assert ast.unparse(final_mma) == "mma_emitter.mma(up_matrix, down_matrix1, out_local)"
    tail = [barrier, final_mma, up_copy, down_copy]
    if version in (737, 738):
        fragment = "next_up" if version == 737 else "next_down"
        rows = "bt1" if version == 737 else "bh2"
        allocation = ast.parse(f"{fragment} = T.alloc_fragment(({rows}, be2), dtype=dtype)").body[0]
        kernel = next(node for node in ast.walk(clone) if isinstance(node, ast.With))
        out_allocation = next(index for index, node in enumerate(kernel.body)
                              if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "out_local")
        kernel.body.insert(out_allocation + 1, allocation)
        global_copy = copy.deepcopy(up_copy if version == 737 else down_copy)
        global_copy.value.args[1] = ast.Name(id=fragment, ctx=ast.Load())
        local_copy = copy.deepcopy(up_copy if version == 737 else down_copy)
        local_copy.value.args[0] = ast.Name(id=fragment, ctx=ast.Load())
        tail = ([barrier, global_copy, final_mma, local_copy, down_copy] if version == 737 else
                [barrier, global_copy, final_mma, up_copy, local_copy])
    loop.body[-4:] = tail
    expected.body.insert(expected.body.index(expected_functions["_pick_tiles"]), clone)
    selection = builder_selection(expected_functions["_get_stage2"])
    selection.value = ast.IfExp(
        test=ast.parse("num_experts == 32 and hidden == 7168 and intermediate == 2048", mode="eval").body,
        body=ast.Name(id=BUILDERS[version], ctx=ast.Load()), orelse=selection.value)
    assert ast.dump(expected) == ast.dump(TREES[version]), version
    original_guard = positive_guard(FUNCS[720][BUILDERS[720]])
    new_guard = positive_guard(FUNCS[version][BUILDERS[version]])
    assert ast.dump(ast.Module(body=original_guard.body[4:], type_ignores=[])) == ast.dump(
        ast.Module(body=new_guard.body[4:], type_ignores=[]))
    print(f"v{version}: whole-module AST whitelist, unchanged original-function source and terminal PASS")


def evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<index-audit>", "eval"), {"__builtins__": {}}, environment)


def audit_dataflow(version, k_steps):
    builder = FUNCS[version][BUILDERS[version]]
    guard = positive_guard(builder)
    loop = steady_loop(builder)
    loop_index = guard.body.index(loop)
    assert loop_index == 3 and len(guard.body[loop_index + 1:]) == 12
    state, outstanding_reads, products = {}, set(), []
    counts = Counter()
    global_tiles = {"up_logits": [], "down_w": []}
    environment = {"block_start": 256, "bt1": 128, "expert_id": 7, "by": 3,
                   "bh2": 128, "be2": 64, "k": 0}

    def execute(statement, expected_k):
        function = expression_name(statement)
        args = statement.value.args
        if function == "T.copy":
            source, target = args
            target_name = ast.unparse(target)
            if isinstance(source, ast.Subscript):
                kind = ast.unparse(source.value)
                indices = source.slice.elts
                assert kind in global_tiles
                ranges = []
                for index in indices:
                    if isinstance(index, ast.Slice):
                        ranges.append((evaluate(index.lower, environment), evaluate(index.upper, environment)))
                    else:
                        ranges.append(evaluate(index, environment))
                assert ranges[:-1] == ([(256, 384)] if kind == "up_logits" else [7, (384, 512)])
                start, stop = ranges[-1]
                assert stop - start == 64 and start % 64 == 0 and 0 <= start < stop <= k_steps * 64
                tile = start // 64
                global_tiles[kind].append(tile)
                tag = (kind, tile)
                counts["global_reads"] += 1
                if target_name in ("next_up", "next_down"):
                    assert tile == expected_k + 1
                    assert state["up_matrix"] == ("up_logits", expected_k, 3)
                    assert state["down_matrix1"] == ("down_w", expected_k, 3)
                    counts["short_prefetches"] += 1
            else:
                tag = state[ast.unparse(source)]
                counts["fragment_to_shared"] += 1
            if target_name.endswith("_shared"):
                assert target_name not in outstanding_reads, (version, expected_k, target_name)
            state[target_name] = tag
            counts["copies"] += 1
        elif function == "T.clear":
            assert ast.unparse(args[0]) == "out_local" and not products
            counts["clears"] += 1
        elif function in ("mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b"):
            register, shared = (ast.unparse(arg) for arg in args[:2])
            micro = evaluate(args[2], environment)
            assert state[shared][1] == expected_k and micro in range(4)
            state[register] = state[shared] + (micro,)
            outstanding_reads.add(shared)
            counts[function.rsplit("_", 1)[1] + "_loads"] += 1
        elif function == "mma_emitter.mma":
            a, b, c = (ast.unparse(arg) for arg in args)
            assert c == "out_local"
            tag_a, tag_b = state[a], state[b]
            assert tag_a[0] == "up_logits" and tag_b[0] == "down_w"
            assert tag_a[1:] == tag_b[1:] and tag_a[1] == expected_k
            products.append(tag_a[1:])
        elif function == "T.sync_threads":
            assert outstanding_reads == {"up_shared", "down_shared"}
            outstanding_reads.clear()
            counts["barriers"] += 1
        else:
            raise AssertionError(function)

    for statement in guard.body[:loop_index]:
        execute(statement, 0)
    for k in range(k_steps - 1):
        environment["k"] = k
        for statement in loop.body:
            execute(statement, k)
    for statement in guard.body[loop_index + 1:]:
        execute(statement, k_steps - 1)
    assert products == list(itertools.product(range(k_steps), range(4)))
    assert all(tiles == list(range(k_steps)) for tiles in global_tiles.values())
    assert counts["global_reads"] == 2 * k_steps
    assert counts["a_loads"] == counts["b_loads"] == 4 * k_steps
    assert counts["clears"] == 1 and counts["barriers"] == k_steps - 1
    assert counts["copies"] == 2 * k_steps + (k_steps - 1 if version in (737, 738) else 0)
    assert counts["short_prefetches"] == counts["fragment_to_shared"] == (
        k_steps - 1 if version in (737, 738) else 0)


for steps in (1, 2, 3, 32, 64):
    for candidate in FILES:
        audit_dataflow(candidate, steps)
    print(f"K={steps}: exact ordered K16 products, current/next tile tags, WAR barriers, copy bounds PASS")


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.dtype, self.device = shape, dtype, "mock:0"


def run_mock(version, experts, hidden, intermediate, route_dtype, padded=6144, valid=4544, blocks=48):
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
    all_inputs = []
    for repeat in range(2):
        specs = [((padded, hidden), "float16"), ((experts, intermediate, hidden), "float16"),
                 ((experts, intermediate, hidden), "float16"), ((experts, hidden, intermediate), "float16"),
                 ((valid,), route_dtype), ((experts,), "int32"), ((experts + 1,), "int32"),
                 ((experts + 1,), "int32"), ((blocks,), "int32"), ((padded, hidden), "float16")]
        inputs = tuple(Tensor(shape, dtype) for shape, dtype in specs)
        all_inputs.append(inputs)
        namespace["run_kernel"](*inputs)
        assert len(launches) == 2 * (repeat + 1) and len(compiled) == 2 and len(workspaces) == 1
        assert launches[-2][1] == tuple(inputs[index] for index in (0, 1, 2, 5, 7, 8)) + (workspaces[0],)
        assert launches[-1][1] == (workspaces[0],) + tuple(inputs[index] for index in (3, 4, 5, 6, 7, 8, 9))
        assert compiled[1][1][-1] == route_dtype
    assert all(first is not second for first, second in zip(*all_inputs))
    return compiled


for experts, hidden, intermediate in ((1, 512, 256), (8, 512, 256), (16, 2048, 8192),
                                     (32, 7168, 2048), (32, 4096, 2048), (32, 7168, 1024),
                                     (64, 7168, 2048)):
    for dtype in ("float16", "float32"):
        original = run_mock(720, experts, hidden, intermediate, dtype)
        for candidate in (736, 737, 738):
            actual = run_mock(candidate, experts, hidden, intermediate, dtype)
            expected = copy.deepcopy(original)
            if (experts, hidden, intermediate) == (32, 7168, 2048):
                expected[1] = (BUILDERS[candidate], *expected[1][1:])
            assert actual == expected
print("Both route dtypes: exact E32/H7168/I2048-only dispatch; fresh-input two launches and cache behavior PASS")

for candidate in FILES:
    for dtype in ("float16", "float32"):
        for padded, valid, blocks in ((0, 0, 0), (256, 0, 2), (256, 1, 2), (512, 129, 4)):
            run_mock(candidate, 32, 7168, 2048, dtype, padded, valid, blocks)
print("Empty/short-tail host metadata preserves v720 behavior (including zero-grid launch requests) PASS")


def rounded(value, dtype):
    code = "e" if dtype == "float16" else "f"
    return struct.unpack(code, struct.pack(code, value))[0]


class GuardedValues:
    def __init__(self, actual, raw_base, route_dtype, kind):
        self.actual, self.raw_base, self.route_dtype, self.kind = actual, raw_base, route_dtype, kind
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        if self.kind == "route":
            assert self.raw_base <= key < self.raw_base + self.actual, key
            return rounded((key % 19 - 9) / 13, self.route_dtype)
        i, j = key
        assert 0 <= i < self.actual and 0 <= j < 128, key
        return rounded((i - j) / 127, "float32")


class Output:
    def __init__(self):
        self.values = {}

    def __setitem__(self, key, value):
        assert key not in self.values
        self.values[key] = struct.pack("e", rounded(value, "float32"))


for version in FILES:
    builder = FUNCS[version][BUILDERS[version]]
    kernel = next(node for node in ast.walk(builder) if isinstance(node, ast.With))
    metadata = next(node for node in kernel.body if isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]) == "active_k_steps")
    guard = positive_guard(builder)
    epilogue = kernel.body[kernel.body.index(guard) + 1:]
    assert len(epilogue) == 1 and isinstance(epilogue[0], ast.If)
    statement = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(epilogue), type_ignores=[])),
                        "<source-epilogue-audit>", "exec")
    language = types.SimpleNamespace(Parallel=lambda a, b: itertools.product(range(a), range(b)),
                                     if_then_else=lambda predicate, a, b: a if predicate else b,
                                     ceildiv=lambda a, b: (a + b - 1) // b)
    for actual in range(129):
        for dtype in ("float16", "float32"):
            # raw_base=0, actual=0 explicitly represents an empty route array.
            raw_base = 0 if actual == 0 else 37
            route = GuardedValues(actual, raw_base, dtype, "route")
            accumulator = GuardedValues(actual, raw_base, dtype, "accumulator")
            output = Output()
            environment = {"T": language, "actual_rows": actual, "bt1": 128, "bh2": 128,
                           "block_start": 256, "by": 3, "out": output, "out_local": accumulator,
                           "routed_expert_weights": route, "raw_start": raw_base, "token_offset": 0,
                           "intermediate": 2048, "be2": 64}
            active = evaluate(metadata.value, environment)
            assert active == (32 if actual else 0)
            exec(statement, {"__builtins__": {}}, environment)
            assert len(output.values) == 128 * 128
            assert len(route.reads) == len(accumulator.reads) == actual * 128
            assert all(raw_base <= index < raw_base + actual for index in route.reads)
            for i, j in itertools.product(range(128), range(128)):
                expected = 0 if i >= actual else rounded((i - j) / 127, "float32") * rounded(
                    ((raw_base + i) % 19 - 9) / 13, dtype)
                assert output.values[256 + i, 384 + j] == struct.pack("e", rounded(expected, "float32"))
    print(f"v{version}: source epilogue actual_rows=0..128, empty raw array, f16/f32 weights and fp16 bytes PASS")

for version in FILES:
    print(f"v{version} SHA256 {hashlib.sha256(FILES[version].read_bytes()).hexdigest()}")
print("LIMIT: CPU WAR/dataflow model does not prove compiler-inserted RAW barriers or lowered route-load safety.")
print("LIMIT: v720 safe-memory-off raw-load hoisting risk is inherited; these probes do not add v723 bounds fixes.")
print("LIMIT: zero padded tokens still requests two zero-grid launches; no new empty-input fast return is claimed.")
