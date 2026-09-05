"""CPU-only v723 -> v744 source, execution-tag, epilogue and host audit.

No torch/TileLang import or GPU execution. Reuse only frozen host-mock helper
definitions from the v743 CPU audit, not its candidate-specific expectations.
Positive K means the existing supported dimensions, not a new zero-K contract.
"""

import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import re
import types


DATA = Path(__file__).resolve().parents[2]
PATHS = {
    723: DATA / "probe_v723_v720_e32_route_load_bounds.py",
    744: DATA / "probe_v744_v723_e32_stage2_static_k_guard.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
             for v, t in TREE.items()}
BASE = "_moe_stage2_fast_bfrag_prefetch_route_bounds"
CLAMP = "T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))"


def dump(node):
    return ast.dump(node, include_attributes=False)


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def kernel_of(builder):
    found = [n for n in ast.walk(builder) if isinstance(n, ast.With)]
    assert len(found) == 1
    return found[0]


expected = copy.deepcopy(TREE[723])
builder = next(n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == BASE)
kernel = kernel_of(builder)
active = next(n for n in kernel.body if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) == "active_k_steps")
assert ast.unparse(active.value) == "T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)"
kernel.body.remove(active)
guard = next(n for n in kernel.body if isinstance(n, ast.If)
             and ast.unparse(n.test) == "active_k_steps > 0")
guard.test = ast.parse("actual_rows > 0", mode="eval").body
loop = next(n for n in guard.body if isinstance(n, ast.For))
assert ast.unparse(loop.iter) == "range(active_k_steps - 1)"
loop.iter = ast.parse("range(T.ceildiv(intermediate, be2) - 1)", mode="eval").body
assert dump(expected) == dump(TREE[744]), "change outside the three permitted AST edits"

# Check executable text as well: strip only the leading header, then replace
# exactly the three known statements within the single intended builder.
old_builder = ast.get_source_segment(SOURCE[723], FUNCTIONS[723][BASE])
expected_builder = old_builder.replace(
    "            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)\n\n", "")
expected_builder = expected_builder.replace("if active_k_steps > 0:", "if actual_rows > 0:")
expected_builder = expected_builder.replace("range(active_k_steps - 1)", "range(T.ceildiv(intermediate, be2) - 1)")
expected_text = SOURCE[723][SOURCE[723].index("import torch\n"):].replace(old_builder, expected_builder)
assert expected_text == SOURCE[744][SOURCE[744].index("import torch\n"):]
assert set(FUNCTIONS[723]) == set(FUNCTIONS[744])
for name in FUNCTIONS[723]:
    if name != BASE:
        assert ast.get_source_segment(SOURCE[723], FUNCTIONS[723][name]) == ast.get_source_segment(SOURCE[744], FUNCTIONS[744][name])
compile(SOURCE[744], str(PATHS[744]), "exec")
print("Full-module AST/executable text: exactly remove active assignment, direct row guard, static loop; all other code PASS")

KERNELS = {v: kernel_of(FUNCTIONS[v][BASE]) for v in PATHS}
GUARDS = {v: next(n for n in k.body if isinstance(n, ast.If)) for v, k in KERNELS.items()}
for v in PATHS:
    routes = [n for n in ast.walk(KERNELS[v]) if isinstance(n, ast.Subscript)
              and isinstance(n.value, ast.Name) and n.value.id == "routed_expert_weights"]
    assert len(routes) == 2 and all(ast.unparse(n.slice) == CLAMP for n in routes)
    assert len([n for n in ast.walk(FUNCTIONS[v][BASE]) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == "T.Kernel"]) == 1
    allocations = [ast.unparse(n.args[0]) for n in ast.walk(KERNELS[v])
                   if isinstance(n, ast.Call) and ast.unparse(n.func) == "T.alloc_shared"]
    assert allocations == ["(bt1, be2)", "(bh2, be2)"]
assert dump(KERNELS[723].body[-1]) == dump(KERNELS[744].body[-1])
LANGUAGE = types.SimpleNamespace(ceildiv=lambda a, b: (a + b - 1) // b,
                                 if_then_else=lambda p, a, b: a if p else b,
                                 min=min, max=max,
                                 Parallel=lambda a, b: itertools.product(range(a), range(b)))


def evaluate(node, env):
    return eval(compile(ast.Expression(node), "<v744-expression>", "eval"),
                {"__builtins__": {}, "range": range}, env)


def execution_tags(version, rows, steps):
    env = {"T": LANGUAGE, "actual_rows": rows, "intermediate": steps * 64,
           "be2": 64, "bt1": 128, "bh2": 128, "block_start": 512,
           "by": 2, "expert_id": 7}
    if version == 723:
        env["active_k_steps"] = evaluate(active.value, env)
    shared, registers, readers, events = {}, {}, set(), []
    products, cleared = [], False

    def execute(node):
        nonlocal cleared
        if isinstance(node, ast.If):
            for child in node.body if evaluate(node.test, env) else node.orelse:
                execute(child)
            return
        if isinstance(node, ast.For):
            assert ast.unparse(node.iter.func) == "range"
            for k in evaluate(node.iter, env):
                env[node.target.id] = k
                for child in node.body:
                    execute(child)
            return
        assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        call, name = node.value, ast.unparse(node.value.func)
        if name == "T.copy":
            source, destination = call.args
            kind = source.value.id
            indices = [((evaluate(x.lower, env), evaluate(x.upper, env))
                        if isinstance(x, ast.Slice) else evaluate(x, env))
                       for x in source.slice.elts]
            assert indices[:-1] == ([(512, 640)] if kind == "up_logits" else [7, (256, 384)])
            lo, hi = indices[-1]
            assert 0 <= lo < hi <= steps * 64 and hi - lo == 64 and lo % 64 == 0
            target = destination.id
            assert target == ("up_shared" if kind == "up_logits" else "down_shared")
            assert target not in readers, "WAR: overwrite before explicit end-K barrier"
            shared[target] = (kind, lo // 64)
            events.append((name, kind, lo // 64, tuple(indices)))
        elif name == "T.clear":
            assert call.args[0].id == "out_local" and not products
            assert not cleared
            cleared = True
            events.append((name,))
        elif name in ("mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b"):
            register, tile, micro = call.args
            assert tile.id == ("up_shared" if name.endswith("_a") else "down_shared")
            ki = evaluate(micro, env)
            assert 0 <= ki < 4
            registers[register.id] = shared[tile.id] + (ki,)
            readers.add(tile.id)
            events.append((name, register.id, registers[register.id]))
        elif name == "mma_emitter.mma":
            a, b, c = [x.id for x in call.args]
            assert cleared and c == "out_local"
            left, right = registers[a], registers[b]
            assert left[0] == "up_logits" and right[0] == "down_w" and left[1:] == right[1:]
            assert left[1:] == (len(products) // 4, len(products) % 4)
            products.append(left[1:])
            events.append((name, left[1:]))
        elif name == "T.sync_threads":
            assert readers == {"up_shared", "down_shared"} and len(products) % 4 == 0
            readers.clear()
            events.append((name, len(products)))
        else:
            raise AssertionError(name)

    execute(GUARDS[version])
    if rows:
        assert products == list(itertools.product(range(steps), range(4)))
        assert [e[1] for e in events if e[0] == "T.sync_threads"] == list(range(4, steps * 4, 4))
    else:
        assert not events and not products and not cleared
    return events


for rows, steps in itertools.product(range(129), (1, 2, 32)):
    assert execution_tags(723, rows, steps) == execution_tags(744, rows, steps)
print("Rows0..128 x K1/2/32: identical M128 copy/clear/LDS/MMA/explicit-barrier trace, no zero-row input loads PASS")

# Do not silently generalize positive-K equivalence to intermediate=0. The
# original guard is false but the direct positive-row guard is true in that
# unsupported combination. No GPU or numerical equivalence is claimed there.
zero_k = {"T": LANGUAGE, "actual_rows": 1, "intermediate": 0, "be2": 64}
zero_k["active_k_steps"] = evaluate(active.value, zero_k)
assert not evaluate(GUARDS[723].test, zero_k) and evaluate(GUARDS[744].test, zero_k)
print("LIMIT confirmed: positive rows with intermediate=0 are not equivalent; no extension of the baseline shape contract")


class Value:
    def __init__(self, row, column):
        self.row, self.column = row, column

    def __mul__(self, route_index):
        return self.row, self.column, route_index


class Accumulator:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        i, j = index
        assert 0 <= i < self.rows and 0 <= j < 128, "invalid/uninitialized C read"
        return Value(i, j)


class Routes:
    def __init__(self, raw_base, rows, total):
        self.raw_base, self.rows, self.total = raw_base, rows, total
        self.reads = []

    def __getitem__(self, index):
        assert 0 <= index < self.total
        assert self.raw_base <= index < self.raw_base + self.rows
        self.reads.append(index)
        return index


class Output:
    def __init__(self):
        self.values = {}

    def __setitem__(self, index, value):
        assert index not in self.values
        self.values[index] = value


epi = compile(module([copy.deepcopy(KERNELS[744].body[-1])]), "<v744-actual-epilogue>", "exec")
for rows, raw_base, is_last in itertools.product(range(129), (0, 293), (False, True)):
    total = raw_base + rows + (0 if is_last else 13)
    route, out = Routes(raw_base, rows, total), Output()
    env = {"T": LANGUAGE, "actual_rows": rows, "bt1": 128, "bh2": 128,
           "block_start": 512, "by": 2, "raw_start": raw_base, "token_offset": 0,
           "total_valid_tokens": total, "routed_expert_weights": route,
           "out_local": Accumulator(rows), "out": out}
    exec(epi, {"__builtins__": {}}, env)
    assert len(out.values) == 128 * 128 and len(route.reads) == rows * 128
    for i, j in itertools.product(range(128), range(128)):
        assert out.values[512 + i, 256 + j] == ((i, j, raw_base + i) if i < rows else 0)
    if total:
        for i in range(128):
            address = evaluate(ast.parse(CLAMP, mode="eval").body, {**env, "i": i})
            assert 0 <= address < total
            if i < rows:
                assert address == raw_base + i
print("Unchanged actual epilogue: complete output ownership, valid same-row raw clamp, zero padding and zero-row no-C/no-route reads PASS")

# Execute only the reusable host instrumentation definitions, not the v743
# audit's tests or GPU code. Both versions under test use their own host AST.
helper = DATA / "bench_records/v743/audit_v743_cpu.py"
helper_tree = ast.parse(helper.read_text(encoding="utf-8"))
helper_nodes = [copy.deepcopy(n) for n in helper_tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("Tensor", "host_mock")]
assert {n.name for n in helper_nodes} == {"Tensor", "host_mock"}
exec(compile(module(helper_nodes), str(helper), "exec"), globals())
host_mock = globals()["host_mock"]
shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192),
          (32, 7168, 2048), (32, 4096, 2048), (32, 7168, 1024), (64, 7168, 2048))
host_cases = 0
for (experts, hidden, intermediate), valid, padded, blocks, dtype in itertools.product(
        shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    args = experts, hidden, intermediate, valid, padded, blocks, dtype
    assert host_mock(723, *args) == host_mock(744, *args)
    host_cases += 1
print(f"Host {host_cases} cases x two fresh calls: identical dispatch, arguments, 0/1/2 launch requests and allocation/JIT-only reuse PASS")
print("LIMIT: mock empty-route/output checks do not prove GPU zero-grid behavior or add support for malformed metadata/zero dimensions.")
print("LIMIT: source WAR proof does not verify inserted RAW barriers, generated source/ISA, physical resources, numerical or bitwise equality.")
print(f"v744 SHA256 {hashlib.sha256(PATHS[744].read_bytes()).hexdigest()}")


def generated_source(path):
    text = path.read_text(encoding="utf-8")
    meta = json.loads(next(line for line in text.splitlines() if line.startswith("{")))
    source = text[text.index("SOURCE_BEGIN "):].split("\n", 1)[1][:meta["source_characters"]]
    assert hashlib.sha256(source.encode()).hexdigest() == meta["source_sha256"]
    return meta, source


records = {
    723: DATA / "bench_records/v743/codex_e32_723_743_codegen_fp32.log",
    744: DATA / "bench_records/v744/codex_e32_744_codegen_fp32.log",
}
if all(path.exists() for path in records.values()):
    generated = {v: generated_source(path) for v, path in records.items()}
    old_meta, old_cpp = generated[723]
    new_meta, new_cpp = generated[744]
    assert old_meta["shared_offsets"] == new_meta["shared_offsets"]
    assert old_meta["local_array_declarations"] == new_meta["local_array_declarations"]
    assert old_meta["route_load_source_lines"] == new_meta["route_load_source_lines"]
    assert old_meta["static_syncthreads_sites"] == new_meta["static_syncthreads_sites"] == 3
    assert new_meta["outer_loop_headers"] == ["for (int k = 0; k < 31; ++k) {"]
    assert "condval" not in new_cpp
    assert new_cpp.count("if (0 < ((group_size + padded_start) - (((int)blockIdx.x) * 128))) {") == 2

    def lds_lines(source):
        return [line.strip() for line in source.splitlines()
                if line.strip().startswith(("up_matrix[", "down_matrix0[", "down_matrix1["))]

    def mma_statements(source):
        lines, found = source.splitlines(), []
        for i, line in enumerate(lines):
            if "__builtin_mxc_mma_16x16x16f16" in line:
                statement = line.strip()
                while not statement.endswith(";"):
                    i += 1
                    statement += lines[i].strip()
                found.append(statement)
        return found

    assert len(lds_lines(old_cpp)) == 16 and lds_lines(old_cpp) == lds_lines(new_cpp)
    assert len(mma_statements(old_cpp)) == 8 and mma_statements(old_cpp) == mma_statements(new_cpp)
    epi_marker = "  if (max(0, min(128, "
    old_epi, new_epi = (source[source.index(epi_marker):] for source in (old_cpp, new_cpp))
    assert old_epi.replace("broadcast_var_3", "broadcast_var_1") == new_epi

    def normalize(expression):
        return ast.parse(re.sub(r"\(int64_t\)|\(int\)", "", expression), mode="eval")

    for kind, target in (("up_logits", "up_shared"), ("down_w", "down_shared")):
        old_load = next(line.strip() for line in old_cpp.splitlines()
                        if f"{kind} + " in line and "condval_" in line)
        new_load = next(line.strip() for line in new_cpp.splitlines()
                        if f"{kind} + " in line and ("k * 64" in line or "(int64_t)k" in line))
        old_expr, new_expr = (line.split(f"{kind} + ", 1)[1][:-2] for line in (old_load, new_load))
        # Redundant parentheses disappear in the Python AST. Removing only C
        # integer casts permits a structural comparison of the index algebra.
        assert dump(normalize(old_expr)) == dump(normalize(new_expr))
        temporary = old_load.split(" = ", 1)[0]
        old_store = next(line.strip() for line in old_cpp.splitlines()
                         if target in line and line.strip().endswith(f"= {temporary};"))
        assert old_store.split(" = ", 1)[0] == new_load.split(" = ", 1)[0]
    print("Captured FP32 codegen: same shared/private declarations, 16 LDS, 8 MMA statements, route loads and epilogue; next-copy index algebra matches PASS")
    print("Captured FP32 codegen: direct row guards, k<31, no condval, same 3 barrier sites; no FP16 or machine-code/resource claim")
else:
    print("Generated-source audit SKIPPED: archived FP32 logs not locally available; CPU source checks above still ran")
