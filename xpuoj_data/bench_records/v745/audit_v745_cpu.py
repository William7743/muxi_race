"""CPU-only v745 source isolation, dispatch and current-K tag audit.

Does not import torch/TileLang or execute GPU code. Run from any directory.
Symbolic arithmetic is not a numerical MMA or compiler-layout proof.
"""

import ast
import copy
import hashlib
import itertools
import types
from pathlib import Path

D = Path(__file__).resolve().parents[2]
paths = {
    743: D / "probe_v743_v723_e32_stage2_runtime_m64.py",
    745: D / "probe_v745_v743_e32_stage1_runtime_m64.py",
}
SOURCE = {v: p.read_text(encoding="utf-8") for v, p in paths.items()}
TREE = {v: ast.parse(s) for v, s in SOURCE.items()}
FUNCTIONS = {
    v: {n.name: n for n in t.body if isinstance(n, ast.FunctionDef)}
    for v, t in TREE.items()
}
BASE = "_moe_stage1_prefetch_giu_merge"
NEW = "_moe_stage1_runtime_m64_giu_merge"


def dump(node):
    return ast.dump(node, include_attributes=False)


def module(nodes):
    return ast.Module(body=nodes, type_ignores=[])


old = FUNCTIONS[743][BASE]
new = FUNCTIONS[745][NEW]
kw = next(n for n in ast.walk(new) if isinstance(n, ast.With))
ow = next(n for n in ast.walk(old) if isinstance(n, ast.With))
branch = kw.body[-1]
assert isinstance(branch, ast.If) and ast.unparse(branch.test) == "actual_rows > tail_m"
assert (
    len(branch.orelse) == 1
    and ast.unparse(branch.orelse[0].test) == "actual_rows > 0"
    and not branch.orelse[0].orelse
)
assert dump(module(branch.body)) == dump(module(ow.body[-4:]))
print("Full branch clearG/clearU/compute/epilogue AST identical to743 PASS")
tail = branch.orelse[0].body


class NormalizeTail(ast.NodeTransformer):
    views = 0

    def visit_Subscript(self, n):
        if ast.unparse(n) == "input_shared[0:tail_m, 0:bh1]":
            self.views += 1
            return ast.Name(id="input_shared", ctx=ast.Load())
        return self.generic_visit(n)

    def visit_Name(self, n):
        n.id = {
            "tail_gate_local": "gate_local",
            "tail_up_local": "up_local",
            "tail_m": "bt1",
        }.get(n.id, n.id)
        return n


norm = NormalizeTail()
normalized = norm.visit(module(copy.deepcopy(tail)))
assert norm.views == 6
assert dump(normalized) == dump(module(ow.body[-4:]))
print(
    "Tail only changes2 C names, row bound64,2 input copy regions +4 GEMM A views PASS"
)
expected = copy.deepcopy(old)
expected.name = NEW
expected.body.insert(1, ast.parse("tail_m = 64").body[0])
ew = next(n for n in ast.walk(expected) if isinstance(n, ast.With))
allocation_end = (
    next(
        i
        for i, n in enumerate(ew.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "up_local"
    )
    + 1
)
ew.body[allocation_end:allocation_end] = ast.parse(
    "tail_gate_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)\ntail_up_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)"
).body
ew.body[-4:] = [copy.deepcopy(branch)]
assert dump(expected) == dump(new)
assert (
    sum(
        isinstance(n, ast.Call) and ast.unparse(n.func) == "T.Kernel"
        for n in ast.walk(new)
    )
    == 1
)
assert (
    sum(
        isinstance(n, ast.Call) and ast.unparse(n.func) == "T.alloc_shared"
        for n in ast.walk(new)
    )
    == 2
)
print(
    "New builder onlydeclares tail_m/two C fragments + replaces branch; all params/pass/layout/cache unchanged PASS"
)
target = "hidden == 7168 and intermediate == 2048 and total_padded_tokens > 0 and num_blocks_m > 0"
ds = copy.deepcopy(FUNCTIONS[743]["_get_stage1"])


class Dispatch(ast.NodeTransformer):
    count = 0

    def visit_IfExp(self, n):
        if ast.unparse(n.test) == "num_experts == 32":
            assert ast.unparse(n.body) == BASE
            n.body = ast.IfExp(
                test=ast.parse(target, mode="eval").body,
                body=ast.Name(id=NEW, ctx=ast.Load()),
                orelse=n.body,
            )
            self.count += 1
            return n
        return self.generic_visit(n)


dt = Dispatch()
dt.visit(ds)
assert dt.count == 1 and dump(ds) == dump(FUNCTIONS[745]["_get_stage1"])
rest = copy.deepcopy(TREE[745])
rest.body = [
    n for n in rest.body if not (isinstance(n, ast.FunctionDef) and n.name == NEW)
]
idx = next(
    i
    for i, n in enumerate(rest.body)
    if isinstance(n, ast.FunctionDef) and n.name == "_get_stage1"
)
rest.body[idx] = copy.deepcopy(FUNCTIONS[743]["_get_stage1"])
assert dump(rest) == dump(TREE[743])
for name, f in FUNCTIONS[743].items():
    if name == "_get_stage1":
        continue
    other = FUNCTIONS[745][name]

    def segment(v, node):
        start = min([node.lineno] + [n.lineno for n in node.decorator_list])
        return "\n".join(SOURCE[v].splitlines()[start - 1 : node.end_lineno])

    assert segment(743, f) == segment(745, other), name
print(
    "Whole module AST isolation + byte-identical every original function except_get_stage1 PASS"
)
# Reuse only the independently written host-mock definitions, without running
# or changing the historical audit's top-level tests.
audit = ast.parse((D / "bench_records/v743/audit_v743_cpu.py").read_text())
defs = [
    n
    for n in audit.body
    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    and n.name in {"Tensor", "host_mock"}
]
exec(compile(module(defs), "<reused-host-mock>", "exec"), globals())
host_mock = globals()["host_mock"]
count = 0
shapes = (
    (1, 512, 256),
    (8, 7168, 2048),
    (16, 2048, 8192),
    (32, 7168, 2048),
    (32, 4096, 2048),
    (32, 7168, 1024),
    (64, 7168, 2048),
)
for (e, h, i), valid, padded, blocks, dtype in itertools.product(
    shapes, (0, 1, 129), (0, 256), (0, 2), ("float16", "float32")
):
    a = host_mock(743, e, h, i, valid, padded, blocks, dtype)
    b = host_mock(745, e, h, i, valid, padded, blocks, dtype)
    expected = copy.deepcopy(a)
    if (e, h, i) == (32, 7168, 2048) and valid > 0 and padded > 0 and blocks > 0:
        assert expected[0][0] == BASE
        expected[0] = (NEW, expected[0][1])
    assert b == expected
    count += 1
print(
    "Host",
    count,
    "combinations x2 fresh inputs, all two-launch/fallback/zero cases PASS",
)
for v, p in paths.items():
    print(v, hashlib.sha256(p.read_bytes()).hexdigest())


# Execute the actual selected Python AST on shape/tag buffers. GEMM records
# reduction-tile tags, not numerical GPU MMA; no torch/TileLang is imported.
active_rows = 0
steps_now = 0


class View:
    def __init__(self, buf, selectors):
        self.buf = buf
        self.shape = []
        self.origins = []
        for size, selector in zip(buf.shape, selectors):
            if isinstance(selector, slice):
                start = 0 if selector.start is None else selector.start
                stop = size if selector.stop is None else selector.stop
                assert selector.step is None and 0 <= start <= stop <= size
                self.shape.append(stop - start)
                self.origins.append(start)
            else:
                assert 0 <= selector < size
        self.shape = tuple(self.shape)
        self.selectors = selectors


class Buffer:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape
        self.tag = None
        self.ks = []
        self.reads = set()
        self.writes = {}

    def __getitem__(self, selectors):
        selectors = selectors if isinstance(selectors, tuple) else (selectors,)
        assert len(selectors) == len(self.shape)
        if any(isinstance(x, slice) for x in selectors):
            return View(self, selectors)
        assert "local" in self.name
        r, c = selectors
        assert 0 <= r < active_rows and 0 <= c < 128
        assert self.name.startswith("tail_") == (active_rows <= 64)
        assert self.ks == list(range(steps_now))
        self.reads.add(selectors)
        return (-0.25 + r / 512) if "gate" in self.name else (0.25 + c / 256)

    def __setitem__(self, selectors, value):
        r, c = selectors
        assert (
            self.name == "up_logits" and 256 <= r < 256 + active_rows and 256 <= c < 384
        )
        assert selectors not in self.writes
        self.writes[selectors] = value


class Lang:
    def __init__(self):
        self.copies = []
        self.mmas = []
        self.syncs = 0

    def clear(self, c):
        assert c.name.startswith("tail_") == (active_rows <= 64)
        c.ks = []

    def copy(self, source, dest, **kwargs):
        d = dest.buf if isinstance(dest, View) else dest
        dshape = dest.shape
        if isinstance(source, View):
            assert source.buf.name in {"stacked_expert_tokens", "gate_w", "up_w"}
            assert source.shape[-1] == 64 and source.origins[-1] % 64 == 0
            tag = (source.buf.name, source.origins[-1] // 64, source.shape)
        else:
            assert source.name == "up_prefetch" and source.tag[0] == "up_w"
            tag = source.tag
        assert tag[2] == dshape
        d.tag = tag
        self.copies.append((tag[0], tag[1], d.name, dshape))

    def gemm(self, a, b, c, **kwargs):
        ap = a.buf if isinstance(a, View) else a
        ashape = a.shape
        m = 64 if active_rows <= 64 else 128
        assert ashape == (m, 64) and b.shape == (128, 64) and c.shape == (m, 128)
        assert ap.tag[0] == "stacked_expert_tokens" and ap.tag[2] == (m, 64)
        assert b.tag[0] == ("gate_w" if "gate" in c.name else "up_w")
        assert ap.tag[1] == b.tag[1] == len(c.ks)
        assert c.name.startswith("tail_") == (active_rows <= 64)
        assert kwargs == {"transpose_B": True, "policy": "Square", "k_pack": 2}
        c.ks.append(ap.tag[1])
        self.mmas.append((c.name, ap.tag[1]))

    def sync_threads(self):
        self.syncs += 1

    @staticmethod
    def Parallel(*shape):
        return itertools.product(*(range(s) for s in shape))

    exp2 = staticmethod(lambda x: 2**x)
    GemmWarpPolicy = types.SimpleNamespace(Square="Square")


program = compile(
    ast.fix_missing_locations(module([copy.deepcopy(branch)])),
    "<v745 source branch>",
    "exec",
)
cases = 0
for active_rows, steps_now in itertools.product(range(129), (1, 2, 112)):
    T = Lang()
    env = {
        "T": T,
        "actual_rows": active_rows,
        "tail_m": 64,
        "bt1": 128,
        "bh1": 64,
        "be1": 128,
        "k_steps": steps_now,
        "gu_k_pack": 2,
        "scale": 1.44269504,
        "block_start": 256,
        "expert_id": 3,
        "by": 2,
    }
    shapes = {
        "stacked_expert_tokens": (512, steps_now * 64),
        "gate_w": (32, 2048, steps_now * 64),
        "up_w": (32, 2048, steps_now * 64),
        "input_shared": (128, 64),
        "weight_shared": (128, 64),
        "up_prefetch": (128, 64),
        "gate_local": (128, 128),
        "up_local": (128, 128),
        "tail_gate_local": (64, 128),
        "tail_up_local": (64, 128),
        "up_logits": (512, 2048),
    }
    env.update({name: Buffer(name, shape) for name, shape in shapes.items()})
    exec(program, {"__builtins__": {"range": range}}, env)
    assert len(T.copies) == (steps_now * 4 if active_rows else 0)
    assert len(T.mmas) == (steps_now * 2 if active_rows else 0)
    assert T.syncs == (steps_now * 2 - 1 if active_rows else 0)
    expected = []
    prefix = "tail_" if active_rows <= 64 else ""
    if active_rows:
        for k in range(steps_now):
            expected.extend([(prefix + "gate_local", k), (prefix + "up_local", k)])
    assert T.mmas == expected
    out = env["up_logits"].writes
    assert set(out) == {
        (256 + r, 256 + c) for r in range(active_rows) for c in range(128)
    }
    for (r, c), value in out.items():
        g = -0.25 + (r - 256) / 512
        u = 0.25 + (c - 256) / 256
        assert value == u * (g * (1.0 / (1.0 + 2 ** (-g * 1.44269504))))
    cases += 1
print(
    "Actual branch AST symbolic",
    cases,
    "cases: rows0..128 x Ksteps1/2/112; GIU current-K tags, barriers, fresh fragments, valid-only SwiGLU writes PASS",
)
full = {
    r * 64 + ((c // 4) ^ (r % 16)) * 4 + c % 4
    for r, c in itertools.product(range(128), range(64))
}
tail = {
    r * 64 + ((c // 4) ^ (r % 16)) * 4 + c % 4
    for r, c in itertools.product(range(64), range(64))
}
assert full == set(range(8192)) and tail == set(range(4096))
print(
    "Declared shared vec4 layout prefix64 has same stride, unique offsets0..4095 within A128x64 PASS"
)
print(
    "LIMIT: symbolic checks do not execute T.gemm lowering, compiler auto RAW barriers, real numerical MMA or physical resource allocation."
)
