"""CPU source/AST, K-tag lifetime and host-dispatch audit for v739/v740.

No torch/TileLang installation, kernel compilation or GPU execution is needed.
Only Tensor/run_mock definitions are reused from the frozen v736-v738 audit;
its top-level checks are not executed. All transformation and K-flow checks here
are independent. Run this file from any directory with ordinary Python.
"""

import ast
from collections import Counter
import copy
import hashlib
import itertools
from pathlib import Path
import sys
import types


DATA = Path(__file__).resolve().parents[2]
FILES = {
    720: DATA / "probe_v720_v719_e16_stage2_bfrag_only.py",
    737: DATA / "probe_v737_v720_e32_stage2_short_up_prefetch.py",
    738: DATA / "probe_v738_v720_e32_stage2_short_down_prefetch.py",
    739: DATA / "probe_v739_v737_e32_stage2_short_up_late_barrier.py",
    740: DATA / "probe_v740_v738_e32_stage2_short_down_late_barrier.py",
}
BUILDERS = {
    737: "_moe_stage2_fast_bfrag_tail_up_prefetch",
    738: "_moe_stage2_fast_bfrag_tail_down_prefetch",
    739: "_moe_stage2_fast_bfrag_tail_up_prefetch_late_barrier",
    740: "_moe_stage2_fast_bfrag_tail_down_prefetch_late_barrier",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {
    version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for version, tree in TREES.items()
}


def dump(node):
    return ast.dump(node, include_attributes=False)


def segment(version, name):
    node = FUNCS[version][name]
    start = min([node.lineno] + [item.lineno for item in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[start - 1:node.end_lineno])


def guard_of(builder):
    return next(node for node in ast.walk(builder)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "active_k_steps > 0")


def loop_of(builder):
    return next(node for node in guard_of(builder).body if isinstance(node, ast.For))


def call_name(statement):
    assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    return ast.unparse(statement.value.func)


class RenameBuilder(ast.NodeTransformer):
    def __init__(self, old, new):
        self.old, self.new = old, new
        self.definitions = self.references = 0

    def visit_FunctionDef(self, node):
        if node.name == self.old:
            node.name = self.new
            self.definitions += 1
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == self.old:
            node.id = self.new
            self.references += 1
        return node


for parent, candidate in ((737, 739), (738, 740)):
    old_name, new_name = BUILDERS[parent], BUILDERS[candidate]
    expected = copy.deepcopy(TREES[parent])
    builder = next(node for node in expected.body
                   if isinstance(node, ast.FunctionDef) and node.name == old_name)
    loop = loop_of(builder)
    assert [call_name(node) for node in loop.body[-5:]] == [
        "T.sync_threads", "T.copy", "mma_emitter.mma", "T.copy", "T.copy"]
    assert ast.unparse(loop.body[-6]) == "mma_emitter.ldmatrix_a(up_matrix, up_shared, 3)"
    barrier, prefetch, mma, up_store, down_store = loop.body[-5:]
    assert ast.unparse(mma) == "mma_emitter.mma(up_matrix, down_matrix1, out_local)"
    loop.body[-5:] = [prefetch, mma, barrier, up_store, down_store]
    rename = RenameBuilder(old_name, new_name)
    expected = rename.visit(expected)
    assert (rename.definitions, rename.references) == (1, 1)
    assert dump(expected) == dump(TREES[candidate]), candidate

    # Independently perform the exact line movement on the original source.
    # Only the leading comment header and LF/CRLF are excluded from this check.
    parent_loop = loop_of(FUNCS[parent][old_name])
    old_tail = parent_loop.body[-5:]
    lines = SOURCES[parent].splitlines(keepends=True)
    old_text = "".join(lines[old_tail[0].lineno - 1:old_tail[-1].end_lineno])
    reordered_text = "".join(
        "".join(lines[old_tail[index].lineno - 1:old_tail[index].end_lineno])
        for index in (1, 2, 0, 3, 4))
    expected_text = "".join(lines[TREES[parent].body[0].lineno - 1:])
    assert expected_text.count(old_text) == 1 and expected_text.count(old_name) == 2
    expected_text = expected_text.replace(old_text, reordered_text).replace(old_name, new_name)
    candidate_text = "".join(SOURCES[candidate].splitlines(keepends=True)[
        TREES[candidate].body[0].lineno - 1:])
    assert expected_text == candidate_text, (candidate, "source beyond leading header")

    assert set(FUNCS[candidate]) == (set(FUNCS[parent]) - {old_name}) | {new_name}
    for name in FUNCS[720]:
        if name != "_get_stage2":
            assert segment(candidate, name) == segment(720, name), (candidate, name)
    for name in FUNCS[parent]:
        if name not in (old_name, "_get_stage2"):
            assert segment(candidate, name) == segment(parent, name), (candidate, name)
    parent_guard = guard_of(FUNCS[parent][old_name])
    new_guard = guard_of(FUNCS[candidate][new_name])
    assert dump(ast.Module(body=parent_guard.body[4:], type_ignores=[])) == dump(
        ast.Module(body=new_guard.body[4:], type_ignores=[]))
    print(f"v{parent}->v{candidate}: whole-source and AST whitelist, original builders, terminal PASS")


def evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<K-tag-audit>", "eval"),
                {"__builtins__": {}}, environment)


def audit_lifetimes(version, steps):
    guard = guard_of(FUNCS[version][BUILDERS[version]])
    loop = loop_of(FUNCS[version][BUILDERS[version]])
    assert guard.body.index(loop) == 3 and len(guard.body[4:]) == 12
    assert [call_name(node) for node in loop.body[-5:]] == [
        "T.copy", "mma_emitter.mma", "T.sync_threads", "T.copy", "T.copy"]
    state, pending_fragment, shared_readers = {}, {}, set()
    products, shared_writes = [], []
    global_reads = {"up_logits": [], "down_w": []}
    counts = Counter()
    env = {"block_start": 256, "bt1": 128, "expert_id": 7, "by": 3,
           "bh2": 128, "be2": 64, "k": 0}

    def execute(statement, k):
        name = call_name(statement)
        args = statement.value.args
        if name == "T.copy":
            source, target = args
            target_name = ast.unparse(target)
            if isinstance(source, ast.Subscript):
                kind = ast.unparse(source.value)
                assert kind in global_reads
                indices = []
                for index in source.slice.elts:
                    if isinstance(index, ast.Slice):
                        assert index.step is None
                        indices.append((evaluate(index.lower, env), evaluate(index.upper, env)))
                    else:
                        indices.append(evaluate(index, env))
                assert indices[:-1] == ([(256, 384)] if kind == "up_logits" else [7, (384, 512)])
                start, stop = indices[-1]
                assert stop - start == 64 and start % 64 == 0
                assert 0 <= start < stop <= steps * 64
                tile = start // 64
                tag = (kind, tile)
                global_reads[kind].append(tile)
                if target_name in ("next_up", "next_down"):
                    assert target_name not in pending_fragment
                    assert tile == k + 1 and len(products) == k * 4 + 3
                    assert state["up_matrix"] == ("up_logits", k, 3)
                    assert state["down_matrix1"] == ("down_w", k, 3)
                    assert shared_readers == {"up_shared", "down_shared"}
                    pending_fragment[target_name] = tag
                    counts["prefetch"] += 1
            else:
                source_name = ast.unparse(source)
                assert source_name in ("next_up", "next_down")
                tag = pending_fragment.pop(source_name)
                assert tag == state[source_name] and tag[1] == k + 1
                assert len(products) == (k + 1) * 4
                counts["fragment_store"] += 1
            if target_name.endswith("_shared"):
                assert target_name not in shared_readers, (version, k, target_name)
                expected_kind = "up_logits" if target_name == "up_shared" else "down_w"
                assert tag[0] == expected_kind
                shared_writes.append((target_name, tag[1]))
            state[target_name] = tag
            counts["copy"] += 1
        elif name == "T.clear":
            assert ast.unparse(args[0]) == "out_local" and not products
            counts["clear"] += 1
        elif name in ("mma_emitter.ldmatrix_a", "mma_emitter.ldmatrix_b"):
            register, shared = (ast.unparse(arg) for arg in args[:2])
            micro = evaluate(args[2], env)
            assert state[shared][1] == k and micro in range(4)
            state[register] = state[shared] + (micro,)
            shared_readers.add(shared)
            counts[name.rsplit("_", 1)[1] + "_load"] += 1
        elif name == "mma_emitter.mma":
            a, b, c = (ast.unparse(arg) for arg in args)
            assert c == "out_local"
            a_tag, b_tag = state[a], state[b]
            assert a_tag[0] == "up_logits" and b_tag[0] == "down_w"
            assert a_tag[1:] == b_tag[1:] and a_tag[1] == k
            assert a_tag[1:] == (len(products) // 4, len(products) % 4)
            products.append(a_tag[1:])
        elif name == "T.sync_threads":
            assert shared_readers == {"up_shared", "down_shared"}
            assert len(products) == (k + 1) * 4
            assert len(pending_fragment) == 1
            shared_readers.clear()
            counts["barrier"] += 1
        else:
            raise AssertionError(name)

    for statement in guard.body[:3]:
        execute(statement, 0)
    for k in range(steps - 1):
        env["k"] = k
        for statement in loop.body:
            execute(statement, k)
        assert not pending_fragment
    for statement in guard.body[4:]:
        execute(statement, steps - 1)
    assert not pending_fragment
    assert products == list(itertools.product(range(steps), range(4)))
    assert all(tiles == list(range(steps)) for tiles in global_reads.values())
    assert shared_writes == [(name, k) for k in range(steps)
                             for name in ("up_shared", "down_shared")]
    assert counts == Counter(copy=3 * steps - 1, clear=1, a_load=4 * steps,
                             b_load=4 * steps, prefetch=steps - 1,
                             fragment_store=steps - 1, barrier=steps - 1)


for candidate in (739, 740):
    for steps in (*range(1, 130), 256, 513):
        audit_lifetimes(candidate, steps)
    print(f"v{candidate}: K=1..129,256,513 ordered products/bounds, one-use next fragment, late WAR barrier PASS")


# Reuse only the existing mock definitions, without importing/executing its
# top-level source/dataflow/epilogue audit. No real torch or TileLang is loaded.
mock_path = DATA / "bench_records/v736_v738/audit_v736_v738_cpu.py"
mock_tree = ast.parse(mock_path.read_text(encoding="utf-8"))
mock_nodes = [node for node in mock_tree.body
              if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("Tensor", "run_mock")]
assert len(mock_nodes) == 2
mock_namespace = {"ast": ast, "types": types, "sys": sys, "TREES": TREES, "FILES": FILES}
exec(compile(ast.Module(body=mock_nodes, type_ignores=[]), str(mock_path), "exec"), mock_namespace)
run_mock = mock_namespace["run_mock"]

shapes = ((1, 512, 256), (8, 512, 256), (16, 2048, 8192), (32, 7168, 2048),
          (32, 4096, 2048), (32, 7168, 1024), (64, 7168, 2048))
for parent, candidate in ((737, 739), (738, 740)):
    for experts, hidden, intermediate in shapes:
        for dtype in ("float16", "float32"):
            expected = run_mock(parent, experts, hidden, intermediate, dtype)
            actual = run_mock(candidate, experts, hidden, intermediate, dtype)
            if (experts, hidden, intermediate) == (32, 7168, 2048):
                expected[1] = (BUILDERS[candidate], *expected[1][1:])
            assert actual == expected
    for dtype in ("float16", "float32"):
        for padded, valid, blocks in ((0, 0, 0), (256, 0, 2), (256, 1, 2), (512, 129, 4)):
            actual = run_mock(candidate, 32, 7168, 2048, dtype, padded, valid, blocks)
            assert actual[1][0] == BUILDERS[candidate]
    print(f"v{candidate}: fresh inputs, exact target dispatch, inherited dtypes/pass configs/caches/two launches PASS")

for candidate in FILES:
    print(f"v{candidate} SHA256 {hashlib.sha256(FILES[candidate].read_bytes()).hexdigest()}")
print("LIMIT: this CPU model does not prove generated RAW barriers, actual asynchronous scheduling or performance.")
print("LIMIT: inherited safe-memory-off route-load hoisting is not fixed; no v723 bounds clamp was added.")
print("LIMIT: zero padded tokens preserves the inherited two zero-grid launch requests, not a no-launch fast return.")
