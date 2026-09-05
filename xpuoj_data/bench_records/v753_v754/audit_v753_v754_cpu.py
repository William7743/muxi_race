"""CPU-only exact-source, dispatch, M32/M64/M128 tag and output audit."""

import ast
import copy
import hashlib
import itertools
import struct
import types
from functools import lru_cache
from pathlib import Path

D = Path(__file__).resolve().parents[2]
PATHS = {
    748: D / "probe_v748_v747_e64_stage1_runtime_m64.py",
    753: D / "probe_v753_v748_e64_stage2_runtime_m32.py",
    754: D / "probe_v754_v748_e32_stage2_runtime_m32_m64.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in PATHS.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
             for v, t in TREE.items()}
BASE = "_moe_stage2_runtime_m64_route_bounds"
NEW = "_moe_stage2_e32_runtime_m32_m64_route_bounds"
CLAMP = "T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))"


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


def dump(node):
    return ast.dump(node, include_attributes=False)


def assign(nodes, name):
    return next(n for n in nodes if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == name)


def kernel(fn):
    return next(n for n in ast.walk(fn) if isinstance(n, ast.With))


def expr(source):
    return ast.parse(source, mode="eval").body


def segment(version, name):
    n = FUNCTIONS[version][name]
    start = min([n.lineno] + [d.lineno for d in n.decorator_list])
    return "\n".join(SOURCE[version].splitlines()[start - 1:n.end_lineno])


old = FUNCTIONS[748][BASE]
expected753 = copy.deepcopy(TREE[748])
clone753 = next(n for n in expected753.body if isinstance(n, ast.FunctionDef) and n.name == BASE)
assign(clone753.body, "tail_m").value = expr("32 if num_experts == 64 else 64")
tail_emitter = assign(clone753.body, "tail_mma_emitter").value
next(k for k in tail_emitter.keywords if k.arg == "warp_row_tiles").value = expr("tail_m // 2")
zero_tail = kernel(clone753).body[-1].orelse[0].body[-1]
assert ast.unparse(zero_tail.iter) == "T.Parallel(tail_m, bh2)"
zero_tail.iter.args[0] = expr("bt1 - tail_m")
assert dump(expected753) == dump(TREE[753])
text753 = SOURCE[748][SOURCE[748].index("import torch\n"):]
old_text = segment(748, BASE)
expected_text = old_text.replace("    tail_m = 64\n", "    tail_m = 32 if num_experts == 64 else 64\n")
expected_text = expected_text.replace("        warp_row_tiles=32,\n", "        warp_row_tiles=tail_m // 2,\n")
expected_text = expected_text.replace("for i, j in T.Parallel(tail_m, bh2):\n                    out[block_start + tail_m + i", "for i, j in T.Parallel(bt1 - tail_m, bh2):\n                    out[block_start + tail_m + i")
assert text753.replace(old_text, expected_text) == SOURCE[753][SOURCE[753].index("import torch\n"):]
print("v753 whole AST/text: exactly three expressions changed PASS", flush=True)


class Specialize32(ast.NodeTransformer):
    def visit_Name(self, n):
        values = {"num_experts": 32, "tail_m": 64, "bt1": 128}
        return ast.Constant(values[n.id]) if isinstance(n.ctx, ast.Load) and n.id in values else n

    def generic_visit(self, n):
        n = super().generic_visit(n)
        if isinstance(n, ast.IfExp) and isinstance(n.test, ast.Constant):
            return n.body if n.test.value else n.orelse
        if isinstance(n, (ast.BinOp, ast.Compare)) and not any(isinstance(x, ast.Name) for x in ast.walk(n)):
            value = eval(compile(ast.fix_missing_locations(ast.Expression(n)), "<constant-only>", "eval"), {"__builtins__": {}})
            return ast.Constant(value)
        return n


assert dump(Specialize32().visit(copy.deepcopy(old))) == dump(Specialize32().visit(copy.deepcopy(FUNCTIONS[753][BASE])))
print("v753 E32 compile-time constant specialization matches original builder AST PASS", flush=True)


class TinyNames(ast.NodeTransformer):
    def visit_Name(self, n):
        if n.id.startswith("tail_"):
            n.id = "tiny_" + n.id[5:]
        return n


expected754 = copy.deepcopy(TREE[748])
clone = copy.deepcopy(old)
clone.name = NEW
clone.body[0].value.value = "One E32 Stage2 launch; uniform M128/M64/M32/zero paths with clamped routes."
tiny_decls = [TinyNames().visit(copy.deepcopy(assign(old.body, name)))
              for name in ("tail_m", "tail_mma_emitter", "tail_a_local_size", "tail_b_local_size")]
tiny_decls[0].value = ast.Constant(32)
next(k for k in tiny_decls[1].value.keywords if k.arg == "warp_row_tiles").value = ast.Constant(16)
insert = clone.body.index(assign(clone.body, "tail_b_local_size")) + 1
clone.body[insert:insert] = tiny_decls
ck = kernel(clone)
allocs = [TinyNames().visit(copy.deepcopy(assign(ck.body, n)))
          for n in ("tail_up_matrix", "tail_down_matrix0", "tail_down_matrix1", "tail_out_local")]
insert = ck.body.index(assign(ck.body, "tail_out_local")) + 1
ck.body[insert:insert] = allocs
layout = next(n.value.args[0] for n in ck.body if isinstance(n, ast.Expr)
              and ast.unparse(n.value.func) == "T.annotate_layout")
layout.keys.append(ast.Name(id="tiny_out_local", ctx=ast.Load()))
layout.values.append(expr("tiny_mma_emitter.make_mma_store_layout(tiny_out_local)"))
tail_branch = ck.body[-1].orelse[0]
original_tail = copy.deepcopy(tail_branch.body)
original_zero = copy.deepcopy(tail_branch.orelse)
tiny_body = TinyNames().visit(module(copy.deepcopy(original_tail))).body
tiny_body[-1].iter.args[0] = expr("bt1 - tiny_m")
tail_branch.test = expr("actual_rows > tiny_m")
tail_branch.orelse = [ast.If(test=expr("actual_rows > 0"), body=tiny_body, orelse=original_zero)]
insertion = next(i for i, n in enumerate(expected754.body) if isinstance(n, ast.FunctionDef) and n.name == "_pick_tiles")
expected754.body.insert(insertion, clone)


class SelectE32(ast.NodeTransformer):
    count = 0

    def visit_Name(self, n):
        if n.id == BASE:
            self.count += 1
            return ast.IfExp(test=expr("num_experts == 32"), body=ast.Name(id=NEW, ctx=ast.Load()), orelse=n)
        return n


selector = SelectE32()
selector.visit(next(n for n in expected754.body if isinstance(n, ast.FunctionDef) and n.name == "_get_stage2"))
assert selector.count == 1 and dump(expected754) == dump(TREE[754])
for name in FUNCTIONS[748]:
    if name != "_get_stage2":
        assert segment(748, name) == segment(754, name), name
new_branch = kernel(FUNCTIONS[754][NEW]).body[-1]
assert dump(module(new_branch.body)) == dump(module(kernel(old).body[-1].body))
assert dump(module(new_branch.orelse[0].body)) == dump(module(original_tail))
for v, name in ((753, BASE), (754, NEW)):
    calls = [n for n in ast.walk(FUNCTIONS[v][name]) if isinstance(n, ast.Call)]
    assert sum(ast.unparse(n.func) == "T.Kernel" for n in calls) == 1
    assert [ast.unparse(n.args[0]) for n in calls if ast.unparse(n.func) == "T.alloc_shared"] == ["(bt1, be2)", "(bh2, be2)"]
    routes = [n for n in ast.walk(FUNCTIONS[v][name]) if isinstance(n, ast.Subscript)
              and isinstance(n.value, ast.Name) and n.value.id == "routed_expert_weights"]
    assert len(routes) == (3 if v == 753 else 4)
    assert all(ast.unparse(n.slice) == CLAMP for n in routes)
print("v754 whole AST reconstructed from748; original M128/M64 bodies and all other builders/host text unchanged PASS", flush=True)

# Reuse only the independent host-mock definitions and extend its expected
# empty-input scope to the already-existing v748 exact E64 target.
helper = ast.parse((D / "bench_records/v743/audit_v743_cpu.py").read_text(encoding="utf-8"))
defs = [copy.deepcopy(n) for n in helper.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and n.name in {"Tensor", "host_mock"}]


class EmptyScope(ast.NodeTransformer):
    count = 0

    def visit_Compare(self, n):
        if ast.unparse(n) == "experts == 32":
            self.count += 1
            return expr("experts == 32 or (experts == 64 and hidden == 7168 and intermediate == 2048)")
        return self.generic_visit(n)


scope = EmptyScope()
ns = {"ast": ast, "copy": copy, "types": types, "module": module, "TREE": TREE, "FUNCTIONS": FUNCTIONS}
exec(compile(ast.fix_missing_locations(scope.visit(module(defs))), "<v748-host-oracle>", "exec"), ns)
assert scope.count == 2
shapes = ((1, 512, 256), (8, 7168, 2048), (16, 2048, 8192),
          (32, 7168, 2048), (32, 4096, 2048), (32, 7168, 1024),
          (64, 7168, 2048), (64, 4096, 2048), (64, 7168, 1024))
count = 0
for (e, h, inter), valid, padded, blocks, dtype in itertools.product(shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")):
    expected = ns["host_mock"](748, e, h, inter, valid, padded, blocks, dtype)
    assert ns["host_mock"](753, e, h, inter, valid, padded, blocks, dtype) == expected
    if (e, h, inter) == (32, 7168, 2048) and valid > 0 and padded > 0 and blocks > 0:
        assert expected[-1][0] == BASE
        expected[-1] = (NEW, expected[-1][1])
    assert ns["host_mock"](754, e, h, inter, valid, padded, blocks, dtype) == expected
    count += 1
print(f"Both probes: host {count} combinations x2 fresh inputs; exact targets/other paths/current inputs/empty shortcuts PASS", flush=True)

LANG = types.SimpleNamespace(Parallel=lambda a, b: itertools.product(range(a), range(b)),
                             ceildiv=lambda a, b: (a + b - 1) // b, max=max, min=min)


@lru_cache(None)
def code(n):
    return compile(ast.Expression(n), "<actual-index>", "eval")


@lru_cache(None)
def unparse(n):
    return ast.unparse(n)


def evaluate(n, env):
    return eval(code(n), {"__builtins__": {}, "range": range}, env)


def audit_k(branch, rows, steps, tail_m, three_way):
    m = 128 if rows > tail_m else 32 if three_way and rows <= 32 else tail_m
    prefix = "" if m == 128 else "tiny_" if three_way and m == 32 else "tail_"
    env = {"T": LANG, "actual_rows": rows, "tail_m": tail_m, "tiny_m": 32, "bt1": 128,
           "be2": 64, "bh2": 128, "intermediate": steps * 64, "active_k_steps": steps if rows else 0,
           "block_start": 512, "by": 2, "expert_id": 7}
    shared, regs, readers, copies, products, clears, barriers = {}, {}, set(), [], [], [], []

    def run(n):
        if isinstance(n, ast.If):
            for child in n.body if evaluate(n.test, env) else n.orelse:
                run(child)
        elif isinstance(n, ast.For):
            if unparse(n.iter.func) == "T.Parallel":
                return
            assert unparse(n.iter.func) == "range"
            for value in evaluate(n.iter, env):
                env[n.target.id] = value
                for child in n.body:
                    run(child)
        else:
            assert isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            call = n.value
            name = unparse(call.func)
            if name == "T.clear":
                assert unparse(call.args[0]) == prefix + "out_local"
                clears.append(prefix + "out_local")
            elif name == "T.copy":
                source, dest = call.args
                kind = source.value.id
                indices = [(evaluate(i.lower, env), evaluate(i.upper, env)) if isinstance(i, ast.Slice)
                           else evaluate(i, env) for i in source.slice.elts]
                assert indices[:-1] == ([(512, 512 + m)] if kind == "up_logits" else [7, (256, 384)])
                begin, end = indices[-1]
                assert end - begin == 64 and begin % 64 == 0 and 0 <= begin < end <= steps * 64
                dname = dest.value.id if isinstance(dest, ast.Subscript) else dest.id
                assert dname == ("up_shared" if kind == "up_logits" else "down_shared")
                if kind == "up_logits":
                    assert isinstance(dest, ast.Subscript) == (m != 128)
                    if m != 128:
                        assert unparse(dest) == f"up_shared[0:{'tiny_m' if prefix == 'tiny_' else 'tail_m'}, 0:be2]"
                assert dname not in readers, "unprotected shared overwrite"
                shared[dname] = (kind, begin // 64)
                copies.append((kind, begin // 64))
            elif name.endswith((".ldmatrix_a", ".ldmatrix_b")):
                assert name.split(".")[0] == prefix + "mma_emitter"
                register, smem, micro = call.args
                sname = smem.value.id if isinstance(smem, ast.Subscript) else smem.id
                assert sname == ("up_shared" if name.endswith("_a") else "down_shared")
                if name.endswith("_a"):
                    assert isinstance(smem, ast.Subscript) == (m != 128)
                    if m != 128:
                        assert unparse(smem) == f"up_shared[0:{'tiny_m' if prefix == 'tiny_' else 'tail_m'}, 0:be2]"
                ki = evaluate(micro, env)
                assert 0 <= ki < 4
                regs[register.id] = shared[sname] + (ki,)
                readers.add(sname)
            elif name.endswith(".mma"):
                a, b, out = (arg.id for arg in call.args)
                assert out == prefix + "out_local" and clears == [out]
                assert regs[a][0] == "up_logits" and regs[b][0] == "down_w"
                assert regs[a][1:] == regs[b][1:] == (len(products) // 4, len(products) % 4)
                products.append(regs[a][1:])
            elif name == "T.sync_threads":
                assert readers == {"up_shared", "down_shared"} and len(products) % 4 == 0
                readers.clear()
                barriers.append(len(products))
            else:
                raise AssertionError(name)

    run(branch)
    if rows:
        assert copies == [(kind, k) for k in range(steps) for kind in ("up_logits", "down_w")]
        assert products == list(itertools.product(range(steps), range(4)))
        assert barriers == [4 * k for k in range(1, steps)]
    else:
        assert not copies and not products and not clears and not barriers


class EpilogueOnly(ast.NodeTransformer):
    def visit_Expr(self, n):
        return None

    def visit_For(self, n):
        return None if unparse(n.iter.func) == "range" else self.generic_visit(n)

    def visit_If(self, n):
        n = self.generic_visit(n)
        if not n.body:
            n.body = [ast.Pass()]
        return n


def round_route(x, dtype):
    kind = "e" if dtype == "float16" else "f"
    return struct.unpack(kind, struct.pack(kind, x))[0]


class Route:
    def __init__(self, start, rows, total, dtype):
        self.start, self.rows, self.total, self.dtype, self.reads = start, rows, total, dtype, 0

    def __getitem__(self, i):
        assert self.start <= i < self.start + self.rows and 0 <= i < self.total
        self.reads += 1
        return round_route((i % 19 - 9) / 13, self.dtype)


class Accumulator:
    def __init__(self, name, wanted, rows):
        self.name, self.wanted, self.rows = name, wanted, rows

    def __getitem__(self, pair):
        i, j = pair
        assert self.name == self.wanted and 0 <= i < self.rows and 0 <= j < 128
        return (i - j) / 127


class Output:
    def __init__(self):
        self.values = {}

    def __setitem__(self, pair, value):
        assert pair not in self.values and 512 <= pair[0] < 640 and 256 <= pair[1] < 384
        self.values[pair] = value


k_count = epilogue_count = 0
for version, builder, tail_m, three_way in ((753, BASE, 32, False), (754, NEW, 64, True)):
    branch = kernel(FUNCTIONS[version][builder]).body[-1]
    for rows, steps in itertools.product(range(129), (1, 2, 32)):
        audit_k(branch, rows, steps, tail_m, three_way)
        k_count += 1
    epilogue = compile(ast.fix_missing_locations(EpilogueOnly().visit(module([copy.deepcopy(branch)]))), "<actual-epilogue>", "exec")
    for rows, dtype in itertools.product(range(129), ("float16", "float32")):
        selected = "out_local" if rows > tail_m else "tiny_out_local" if three_way and rows <= 32 else "tail_out_local"
        raw_start, offset = (37, 256) if rows else (0, 0)
        raw_base, total = raw_start + offset, raw_start + offset + rows
        route, out = Route(raw_base, rows, total, dtype), Output()
        env = {"T": LANG, "actual_rows": rows, "tail_m": tail_m, "tiny_m": 32, "bt1": 128, "bh2": 128,
               "active_k_steps": 32 if rows else 0, "block_start": 512, "by": 2,
               "raw_start": raw_start, "token_offset": offset, "total_valid_tokens": total,
               "routed_expert_weights": route, "out": out}
        env.update({name: Accumulator(name, selected, rows) for name in ("out_local", "tail_out_local", "tiny_out_local")})
        exec(epilogue, {"__builtins__": {}}, env)
        assert len(out.values) == 128 * 128 and route.reads == rows * 128
        for i, j in itertools.product(range(128), range(128)):
            wanted = ((i - j) / 127) * round_route(((raw_base + i) % 19 - 9) / 13, dtype) if i < rows else 0
            assert out.values[512 + i, 256 + j] == wanted
        if total:
            for extra, i in itertools.product((0, 13), range(128)):
                address = evaluate(expr(CLAMP), {**env, "total_valid_tokens": total + extra, "i": i})
                assert 0 <= address < total + extra
                if i < rows:
                    assert address == raw_base + i
        epilogue_count += 1
print(f"Actual branch AST: {k_count} rows/K tag cases, exact copy bounds/MMA K16 order/explicit WAR barriers PASS", flush=True)
print(f"Actual epilogue AST: {epilogue_count} rows/dtype cases, each128x128 output once, padded zeros, no invalid raw reads PASS", flush=True)

for m in (32, 64, 128):
    owners = {}
    for thread, row_tile, col_tile, slot in itertools.product(range(256), range(m // 32), range(4), range(4)):
        lane, wm, wn = thread % 64, (thread // 64) % 2, thread // 128
        row = wm * (m // 2) + row_tile * 16 + lane % 16
        col = wn * 64 + col_tile * 16 + lane // 16 * 4 + slot
        assert (row, col) not in owners
        owners[row, col] = (thread, row_tile * 16 + col_tile * 4 + slot)
    assert set(owners) == set(itertools.product(range(m), range(128)))
    footprint = {r * 64 + ((c // 4) ^ (r % 16)) * 4 + c % 4 for r, c in itertools.product(range(m), range(64))}
    assert footprint == set(range(m * 64))
print("Declared 2x2-warp M32/M64/M128 C ownership and vec4 A-prefix footprint PASS", flush=True)
for version, path in PATHS.items():
    compile(SOURCE[version], str(path), "exec")
    print(version, hashlib.sha256(path.read_bytes()).hexdigest())
print("LIMIT: no TileLang import/GPU/MMA rounding, actual view lowering, automatic RAW barriers or physical resource proof. v753 remains untested, not failed.")
