"""Independent CPU-only source, dispatch, K-tag and geometry audit of v743.

No installed torch/TileLang or GPU is used. Symbolic MMA and layout checks are
not a numerical GPU or bitwise-output test. Generated branch/view layout and
compiler-inserted producer/consumer synchronization require a separate audit.
"""

import ast
from collections import Counter
import copy
import hashlib
import itertools
from pathlib import Path
import struct
import types


DATA = Path(__file__).resolve().parents[2]
PATHS = {
    634: DATA / "probe_v634_e32_stage2_m64_bfrag_th256.py",
    723: DATA / "probe_v723_v720_e32_route_load_bounds.py",
    743: DATA / "probe_v743_v723_e32_stage2_runtime_m64.py",
}
SOURCE = {version: path.read_text(encoding="utf-8") for version, path in PATHS.items()}
TREE = {version: ast.parse(text) for version, text in SOURCE.items()}
FUNCTIONS = {version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
             for version, tree in TREE.items()}
BASE = "_moe_stage2_fast_bfrag_prefetch_route_bounds"
NEW = "_moe_stage2_runtime_m64_route_bounds"
OLD_TAIL = "_moe_stage2_e64_tail64"
TARGET = ("hidden == 7168 and intermediate == 2048 and total_valid_tokens > 0 "
          "and total_padded_tokens > 0 and num_blocks_m > 0")
CLAMP = "T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))"


def dump(node):
    return ast.dump(node, include_attributes=False)


def module(statements):
    return ast.Module(body=statements, type_ignores=[])


def segment(version, name):
    node = FUNCTIONS[version][name]
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(SOURCE[version].splitlines()[start - 1:node.end_lineno])


def kernel_of(builder):
    found = [node for node in ast.walk(builder) if isinstance(node, ast.With)]
    assert len(found) == 1
    return found[0]


def assignment(statements, name):
    return next(node for node in statements if isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == name)


TAIL_NAMES = {name: "tail_" + name for name in (
    "mma_emitter", "a_local_size", "b_local_size", "up_matrix",
    "down_matrix0", "down_matrix1", "out_local")}


class TailNames(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = TAIL_NAMES.get(node.id, node.id)
        return node


class TailViewAndClamp(TailNames):
    def visit_Name(self, node):
        node = super().visit_Name(node)
        if node.id == "up_shared":
            return ast.parse("up_shared[0:tail_m, 0:be2]", mode="eval").body
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "routed_expert_weights":
            assert ast.unparse(node.slice) == "raw_start + token_offset + i"
            node.slice = ast.parse(CLAMP, mode="eval").body
        return node


# Construct every permitted new AST node independently from the two historical
# builders; do not whitelist the candidate's own new body as its expectation.
expected = copy.deepcopy(TREE[723])
base_functions = {node.name: node for node in expected.body if isinstance(node, ast.FunctionDef)}
clone = copy.deepcopy(base_functions[BASE])
clone.name = NEW
clone.body[0].value.value = "One E32 Stage2 launch; uniform M128/M64/zero paths with clamped routes."
old_tail = FUNCTIONS[634][OLD_TAIL]
tail_declarations = [ast.parse("tail_m = 64").body[0]] + [
    TailNames().visit(copy.deepcopy(assignment(old_tail.body, name)))
    for name in ("mma_emitter", "a_local_size", "b_local_size")]
insertion = clone.body.index(assignment(clone.body, "b_local_size")) + 1
clone.body[insertion:insertion] = tail_declarations
kernel = kernel_of(clone)
old_tail_kernel = kernel_of(old_tail)
tail_allocations = [TailNames().visit(copy.deepcopy(assignment(old_tail_kernel.body, name)))
                    for name in ("up_matrix", "down_matrix0", "down_matrix1", "out_local")]
insertion = kernel.body.index(assignment(kernel.body, "out_local")) + 1
kernel.body[insertion:insertion] = tail_allocations
annotation = next(node for node in kernel.body if isinstance(node, ast.Expr)
                  and ast.unparse(node.value.func) == "T.annotate_layout")
layouts = annotation.value.args[0]
assert isinstance(layouts, ast.Dict) and len(layouts.keys) == 3
layouts.keys.append(ast.Name(id="tail_out_local", ctx=ast.Load()))
layouts.values.append(ast.parse("tail_mma_emitter.make_mma_store_layout(tail_out_local)", mode="eval").body)
full_body = kernel.body[-2:]
assert ast.unparse(full_body[0].test) == "active_k_steps > 0"
assert ast.unparse(full_body[1].test) == "actual_rows == bt1"
tail_guard = next(node for node in old_tail_kernel.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "actual_rows > 0 and actual_rows <= tail_m")
tail_body = TailViewAndClamp().visit(module(copy.deepcopy(tail_guard.body))).body
zero_body = ast.parse("for i, j in T.Parallel(bt1, bh2):\n    out[block_start + i, by * bh2 + j] = 0").body
selection = ast.If(test=ast.parse("actual_rows > tail_m", mode="eval").body,
                   body=full_body, orelse=[ast.If(test=ast.parse("actual_rows > 0", mode="eval").body,
                                               body=tail_body, orelse=zero_body)])
kernel.body[-2:] = [selection]
expected.body.insert(expected.body.index(base_functions["_pick_tiles"]), clone)


class TargetSelection(ast.NodeTransformer):
    count = 0

    def visit_Name(self, node):
        if node.id == BASE:
            self.count += 1
            return ast.IfExp(test=ast.parse(TARGET, mode="eval").body,
                             body=ast.Name(id=NEW, ctx=ast.Load()), orelse=node)
        return node


replace = TargetSelection()
replace.visit(base_functions["_get_stage2"])
assert replace.count == 1
assert dump(expected) == dump(TREE[743]), "unexpected whole-module AST change"
assert set(FUNCTIONS[743]) == set(FUNCTIONS[723]) | {NEW}
for name in FUNCTIONS[723]:
    if name != "_get_stage2":
        assert segment(743, name) == segment(723, name), name
compile(SOURCE[743], str(PATHS[743]), "exec")
print("Full-module AST reconstructed from 723 + 634: exact allowed change; all original builders/host guards unchanged PASS")

new_kernel = kernel_of(FUNCTIONS[743][NEW])
allocations = [node for node in ast.walk(new_kernel) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == "T.alloc_shared"]
assert [ast.unparse(node.args[0]) for node in allocations] == ["(bt1, be2)", "(bh2, be2)"]
assert dump(new_kernel.items[0].context_expr) == dump(kernel_of(FUNCTIONS[723][BASE]).items[0].context_expr)
assert len([node for node in ast.walk(FUNCTIONS[743][NEW]) if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "T.Kernel"]) == 1
branch = new_kernel.body[-1]
assert ast.unparse(branch.test) == "actual_rows > tail_m"
assert ast.unparse(branch.orelse[0].test) == "actual_rows > 0"
assert len(branch.orelse[0].orelse) == 1
route_nodes = [node for node in ast.walk(branch) if isinstance(node, ast.Subscript)
               and isinstance(node.value, ast.Name) and node.value.id == "routed_expert_weights"]
assert len(route_nodes) == 3 and all(ast.unparse(node.slice) == CLAMP for node in route_nodes)


def evaluate(node, env):
    return eval(compile(ast.Expression(node), "<v743-index>", "eval"),
                {"__builtins__": {}, "range": range}, env)


LANGUAGE = types.SimpleNamespace(Parallel=lambda a, b: itertools.product(range(a), range(b)),
                                 ceildiv=lambda a, b: (a + b - 1) // b,
                                 if_then_else=lambda p, a, b: a if p else b, max=max, min=min)


def audit_k_tags(actual, steps):
    env = {"T": LANGUAGE, "actual_rows": actual, "tail_m": 64, "bt1": 128, "be2": 64,
           "bh2": 128, "intermediate": 64 * steps, "active_k_steps": steps if actual else 0,
           "block_start": 512, "expert_id": 7, "by": 2}
    registers, shared_state, readers = {}, {}, set()
    copies, products, cleared, barriers = [], [], [], []
    m = 128 if actual > 64 else 64
    expected_c = "out_local" if m == 128 else "tail_out_local"

    def execute(statement):
        if isinstance(statement, ast.If):
            for child in statement.body if evaluate(statement.test, env) else statement.orelse:
                execute(child)
            return
        if isinstance(statement, ast.For):
            if ast.unparse(statement.iter.func) == "T.Parallel":
                return  # The actual output AST is executed separately below.
            assert ast.unparse(statement.iter.func) == "range"
            for value in evaluate(statement.iter, env):
                env[statement.target.id] = value
                for child in statement.body:
                    execute(child)
            return
        assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        call = statement.value
        name = ast.unparse(call.func)
        if name == "T.clear":
            assert ast.unparse(call.args[0]) == expected_c and not products
            cleared.append(expected_c)
        elif name == "T.copy":
            source, target = call.args
            assert isinstance(source, ast.Subscript)
            kind = source.value.id
            assert kind in ("up_logits", "down_w")
            indices = [((evaluate(index.lower, env), evaluate(index.upper, env))
                        if isinstance(index, ast.Slice) else evaluate(index, env))
                       for index in source.slice.elts]
            assert indices[:-1] == ([(512, 512 + m)] if kind == "up_logits" else [7, (256, 384)])
            start, stop = indices[-1]
            assert stop - start == 64 and start % 64 == 0 and 0 <= start < stop <= steps * 64
            target_name = target.value.id if isinstance(target, ast.Subscript) else target.id
            assert target_name == ("up_shared" if kind == "up_logits" else "down_shared")
            if isinstance(target, ast.Subscript):
                assert m == 64 and ast.unparse(target) == "up_shared[0:tail_m, 0:be2]"
            else:
                assert kind == "down_w" or m == 128
            assert target_name not in readers, "unprotected shared overwrite"
            shared_state[target_name] = (kind, start // 64)
            copies.append((kind, start // 64))
        elif name.endswith((".ldmatrix_a", ".ldmatrix_b")):
            emitter = name.split(".")[0]
            assert emitter == ("mma_emitter" if m == 128 else "tail_mma_emitter")
            register, shared, micro = call.args
            shared_name = shared.value.id if isinstance(shared, ast.Subscript) else shared.id
            if name.endswith("_a"):
                assert shared_name == "up_shared"
                assert isinstance(shared, ast.Subscript) == (m == 64)
                if m == 64:
                    assert ast.unparse(shared) == "up_shared[0:tail_m, 0:be2]"
            else:
                assert shared_name == "down_shared" and isinstance(shared, ast.Name)
            ki = evaluate(micro, env)
            assert 0 <= ki < 4
            registers[register.id] = shared_state[shared_name] + (ki,)
            readers.add(shared_name)
        elif name.endswith(".mma"):
            a, b, c = (arg.id for arg in call.args)
            assert c == expected_c and cleared == [expected_c]
            left, right = registers[a], registers[b]
            assert left[0] == "up_logits" and right[0] == "down_w" and left[1:] == right[1:]
            assert left[1:] == (len(products) // 4, len(products) % 4)
            products.append(left[1:])
        elif name == "T.sync_threads":
            assert readers == {"up_shared", "down_shared"} and len(products) % 4 == 0
            readers.clear()
            barriers.append(len(products))
        else:
            raise AssertionError(name)

    execute(branch)
    if not actual:
        assert not copies and not products and not cleared and not barriers
    else:
        assert products == list(itertools.product(range(steps), range(4)))
        assert copies == [(kind, k) for k in range(steps) for kind in ("up_logits", "down_w")]
        assert barriers == [4 * k for k in range(1, steps)]


for rows, steps in itertools.product(range(129), (1, 2, 32)):
    audit_k_tags(rows, steps)
print("Rows 0..128 x K=1,2,32: branch choice, current tile/micro-K tags, exact MMA order, view/copy bounds and explicit WAR barriers PASS")


# Geometric ownership model for the declared 2x2 warps, 64 lanes, M16/N16
# microtiles. This is not execution of the actual backend's layout generator.
for m in (64, 128):
    a_owners, b_owners, c_owners = Counter(), Counter(), Counter()
    for thread in range(256):
        lane, wm, wn = thread % 64, (thread // 64) % 2, thread // 128
        for row_tile, col_tile, slot in itertools.product(range(m // 32), range(4), range(4)):
            row = wm * (m // 2) + row_tile * 16 + lane % 16
            col = wn * 64 + col_tile * 16 + (lane // 16) * 4 + slot
            c_owners[row, col] += 1
        for ki, row_tile, slot in itertools.product(range(4), range(m // 32), range(4)):
            row = wm * (m // 2) + row_tile * 16 + lane % 16
            col = ki * 16 + (lane // 16) * 4 + slot
            a_owners[row, col] += 1
            # vec4 swizzle preserves 64-half row stride and the first64 rows.
            physical_half = row * 64 + ((col // 4) ^ (row % 16)) * 4 + col % 4
            assert row * 64 <= physical_half < (row + 1) * 64 <= m * 64
        for ki, col_tile, slot in itertools.product(range(4), range(4), range(4)):
            row = wn * 64 + col_tile * 16 + lane % 16
            col = ki * 16 + (lane // 16) * 4 + slot
            b_owners[row, col] += 1
    assert set(c_owners) == set(itertools.product(range(m), range(128)))
    assert set(c_owners.values()) == {1}
    assert set(a_owners) == set(itertools.product(range(m), range(64)))
    assert set(a_owners.values()) == {2}  # reused by two warp-N planes
    assert set(b_owners) == set(itertools.product(range(128), range(64)))
    assert set(b_owners.values()) == {2}  # reused by two warp-M planes
print("Declared M64/M128 emitter geometry: C coverage once, A/B reuse twice, first64 swizzle footprint stays within shared prefix PASS")


class EpilogueOnly(ast.NodeTransformer):
    def visit_Expr(self, node):
        return None

    def visit_For(self, node):
        if ast.unparse(node.iter.func) == "range":
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if not node.body:
            node.body = [ast.Pass()]
        return node


epilogue = ast.fix_missing_locations(EpilogueOnly().visit(module([copy.deepcopy(branch)])))
epilogue_code = compile(epilogue, "<v743-actual-epilogue>", "exec")


def rounded(value, dtype):
    code = "e" if dtype == "float16" else "f"
    return struct.unpack(code, struct.pack(code, value))[0]


class Route:
    def __init__(self, raw_base, rows, total, dtype):
        self.raw_base, self.rows, self.total, self.dtype = raw_base, rows, total, dtype
        self.reads = []

    def __getitem__(self, index):
        assert 0 <= index < self.total and self.raw_base <= index < self.raw_base + self.rows
        self.reads.append(index)
        return rounded((index % 19 - 9) / 13, self.dtype)


class Accumulator:
    def __init__(self, selected, name, rows):
        self.selected, self.name, self.rows = selected, name, rows
        self.reads = []

    def __getitem__(self, pair):
        i, j = pair
        assert self.name == self.selected and 0 <= i < self.rows and 0 <= j < 128
        self.reads.append(pair)
        return rounded((i - j) / 127, "float32")


class Output:
    def __init__(self):
        self.values = {}

    def __setitem__(self, pair, value):
        assert pair not in self.values and 512 <= pair[0] < 640 and 256 <= pair[1] < 384
        self.values[pair] = struct.pack("e", rounded(value, "float32"))


for rows, dtype, final_group in itertools.product(range(129), ("float16", "float32"), (False, True)):
    raw_start, offset = (0, 0) if rows == 0 and final_group else (37, 256)
    raw_base = raw_start + offset
    total = raw_base + rows + (0 if final_group else 13)
    selected = "out_local" if rows > 64 else "tail_out_local"
    route = Route(raw_base, rows, total, dtype)
    full, tail = (Accumulator(selected, name, rows) for name in ("out_local", "tail_out_local"))
    out = Output()
    env = {"T": LANGUAGE, "actual_rows": rows, "tail_m": 64, "bt1": 128, "bh2": 128,
           "active_k_steps": 32 if rows else 0, "block_start": 512, "by": 2,
           "raw_start": raw_start, "token_offset": offset, "total_valid_tokens": total,
           "out": out, "out_local": full, "tail_out_local": tail, "routed_expert_weights": route}
    exec(epilogue_code, {"__builtins__": {}}, env)
    assert len(out.values) == 128 * 128
    assert len(route.reads) == len(full.reads) + len(tail.reads) == rows * 128
    for i, j in itertools.product(range(128), range(128)):
        value = (rounded((i - j) / 127, "float32") * rounded(((raw_base + i) % 19 - 9) / 13, dtype)
                 if i < rows else 0)
        assert out.values[512 + i, 256 + j] == struct.pack("e", rounded(value, "float32"))
    if total:
        for i in range(128):
            address = evaluate(ast.parse(CLAMP, mode="eval").body, {**env, "i": i})
            assert 0 <= address < total
            if i < rows:
                assert address == raw_base + i
print("Actual source epilogue: rows0..128, raw-empty/last/nonlast group, FP16/FP32 route, each output once, exact padding zeros and same-row clamp PASS")


class Tensor:
    def __init__(self, shape, dtype="float16", device="mock:0"):
        self.shape, self.dtype, self.device = shape, dtype, device


def host_mock(version, experts, hidden, intermediate, valid, padded, blocks, route_dtype):
    built, launched, workspaces = [], [], []

    def empty(shape, *, device, dtype):
        tensor = Tensor(shape, dtype, device)
        workspaces.append(tensor)
        return tensor

    def builder(name):
        def build(*parameters):
            built.append((name, parameters))

            def launch(*inputs):
                launched.append((name, inputs))

            return launch

        return build

    env = {"torch": types.SimpleNamespace(float16="float16", float32="float32", empty=empty),
           "T": types.SimpleNamespace(float16="float16", float32="float32"),
           "_KERNEL_CACHE": {}, "_WORKSPACE_CACHE": {}}
    for name in FUNCTIONS[version]:
        if name.startswith("_moe_"):
            env[name] = builder(name)
    names = {"_pick_tiles", "_get_stage1", "_get_stage2", "_get_workspace", "run_kernel"}
    nodes = [copy.deepcopy(node) for node in TREE[version].body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(module(nodes), "<v743-host-mock>", "exec"), env)
    all_inputs = []
    for repeat in range(2):
        inputs = (Tensor((padded, hidden)), Tensor((experts, intermediate, hidden)),
                  Tensor((experts, intermediate, hidden)), Tensor((experts, hidden, intermediate)),
                  Tensor((valid,), route_dtype), Tensor((experts,), "int32"),
                  Tensor((experts + 1,), "int32"), Tensor((experts + 1,), "int32"),
                  Tensor((blocks,), "int32"), Tensor((padded, hidden)))
        all_inputs.append(inputs)
        before = len(launched)
        env["run_kernel"](*inputs)
        now = launched[before:]
        if experts == 32 and padded == 0:
            assert not now and not built and not workspaces
        else:
            assert len(workspaces) == 1
            workspace = workspaces[0]
            assert now[-1][1] == (workspace,) + tuple(inputs[i] for i in (3, 4, 5, 6, 7, 8, 9))
            if experts == 32 and valid == 0:
                assert len(now) == 1 and now[0][0] == "_moe_stage2_e32_zero_output"
            else:
                assert len(now) == 2
                assert now[0][1] == tuple(inputs[i] for i in (0, 1, 2, 5, 7, 8)) + (workspace,)
            assert len(built) == len(now), "second input unexpectedly rebuilt callables"
            assert built[-1][1][-1] == route_dtype
        assert len(launched) == (repeat + 1) * len(now)
    assert all(first is not second for first, second in zip(*all_inputs))
    return built


shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192), (32, 7168, 2048),
          (32, 4096, 2048), (32, 7168, 1024), (64, 7168, 2048))
host_cases = 0
for (experts, hidden, intermediate), valid, padded, blocks, dtype in itertools.product(
        shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    reference = host_mock(723, experts, hidden, intermediate, valid, padded, blocks, dtype)
    actual = host_mock(743, experts, hidden, intermediate, valid, padded, blocks, dtype)
    expected_builds = copy.deepcopy(reference)
    if experts == 32 and hidden == 7168 and intermediate == 2048 and valid > 0 and padded > 0 and blocks > 0:
        assert expected_builds[-1][0] == BASE
        expected_builds[-1] = (NEW, expected_builds[-1][1])
    assert actual == expected_builds
    host_cases += 1
print(f"Host {host_cases} cases x two fresh inputs: exact target/fallback, empty-output return, empty-route zero-only, JIT/workspace reuse without result replay PASS")
print(f"v743 SHA256 {hashlib.sha256(PATHS[743].read_bytes()).hexdigest()}")
print("LIMIT: symbolic checks do not prove generated M64 view/layout, automatic RAW barriers, hoisted route loads or GPU numerical/bitwise equality.")
print("LIMIT: positive-route clamp has valid addresses; zero-route safety depends on unchanged host zero-kernel dispatch, not clamping an empty array.")
