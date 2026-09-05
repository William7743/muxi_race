"""CPU source/AST and dispatch isolation checks for v741/v742.

Run: python xpuoj_data/bench_records/v741_v742/audit_v741_v742_cpu.py
No GPU/TileLang execution. This proves only the requested source change:
remove one explicit steady barrier and rename its builder/reference. It does
NOT prove that compiler-generated WAR/RAW synchronization remains sufficient.
"""

import ast
import copy
import hashlib
from pathlib import Path
import sys
import types


DATA = Path(__file__).resolve().parents[2]
FILES = {
    739: DATA / "probe_v739_v737_e32_stage2_short_up_late_barrier.py",
    740: DATA / "probe_v740_v738_e32_stage2_short_down_late_barrier.py",
    741: DATA / "probe_v741_v739_e32_stage2_short_up_auto_barrier.py",
    742: DATA / "probe_v742_v740_e32_stage2_short_down_auto_barrier.py",
}
BUILDERS = {
    739: "_moe_stage2_fast_bfrag_tail_up_prefetch_late_barrier",
    740: "_moe_stage2_fast_bfrag_tail_down_prefetch_late_barrier",
    741: "_moe_stage2_fast_bfrag_tail_up_prefetch_auto_barrier",
    742: "_moe_stage2_fast_bfrag_tail_down_prefetch_auto_barrier",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
FUNCS = {version: {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
         for version, tree in TREES.items()}


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


def guard_of(builder):
    return next(node for node in ast.walk(builder)
                if isinstance(node, ast.If) and ast.unparse(node.test) == "active_k_steps > 0")


def loop_of(builder):
    return next(node for node in guard_of(builder).body if isinstance(node, ast.For))


def segment(version, name):
    node = FUNCS[version][name]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return "\n".join(SOURCES[version].splitlines()[first - 1:node.end_lineno])


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


for parent, candidate in ((739, 741), (740, 742)):
    old, new = BUILDERS[parent], BUILDERS[candidate]
    expected = copy.deepcopy(TREES[parent])
    target = next(node for node in expected.body
                  if isinstance(node, ast.FunctionDef) and node.name == old)
    loop = loop_of(target)
    all_barriers = [node for node in ast.walk(target) if call_name(node) == "T.sync_threads"]
    assert len(all_barriers) == 1
    assert [call_name(node) for node in loop.body[-5:]] == [
        "T.copy", "mma_emitter.mma", "T.sync_threads", "T.copy", "T.copy"]
    assert all_barriers[0] is loop.body[-3]
    assert ast.unparse(loop.body[-4]) == "mma_emitter.mma(up_matrix, down_matrix1, out_local)"
    removed = loop.body.pop(-3)
    rename = RenameBuilder(old, new)
    expected = rename.visit(expected)
    assert (rename.definitions, rename.references) == (1, 1)
    assert ast.dump(expected) == ast.dump(TREES[candidate])
    assert not any(call_name(node) == "T.sync_threads" for node in ast.walk(FUNCS[candidate][new]))

    # Independent textual proof: preserve even the trailing whitespace after
    # the imports, allowing only universal-newline decoding and leading header.
    lines = SOURCES[parent].splitlines(keepends=True)
    assert removed.lineno == removed.end_lineno
    removed_line = lines[removed.lineno - 1]
    assert removed_line.strip() == "T.sync_threads()"
    lines.pop(removed.lineno - 1)
    expected_source = "".join(lines)
    expected_source = expected_source[expected_source.index("import torch"):]
    assert expected_source.count(old) == 2
    expected_source = expected_source.replace(old, new)
    candidate_source = SOURCES[candidate][SOURCES[candidate].index("import torch"):]
    assert expected_source == candidate_source

    assert set(FUNCS[candidate]) == (set(FUNCS[parent]) - {old}) | {new}
    for name in FUNCS[parent]:
        if name not in (old, "_get_stage2"):
            assert segment(candidate, name) == segment(parent, name), (candidate, name)
    before, after = guard_of(FUNCS[parent][old]), guard_of(FUNCS[candidate][new])
    assert before.body.index(loop_of(FUNCS[parent][old])) == after.body.index(loop_of(FUNCS[candidate][new])) == 3
    for statements_old, statements_new in ((before.body[:3], after.body[:3]),
                                           (before.body[4:], after.body[4:])):
        assert ast.dump(ast.Module(body=statements_old, type_ignores=[])) == ast.dump(
            ast.Module(body=statements_new, type_ignores=[]))
    assert len(after.body[4:]) == 12
    print(f"v{parent}->v{candidate}: full source/AST only unique steady barrier removal + builder/reference rename PASS")
    print(f"v{candidate}: all other builders/Stage1, prologue, terminal, epilogue, math/copies/pass settings unchanged PASS")

# Import only host mock definitions from the existing audit. Its barrier-lifetime
# tests are intentionally NOT reused: they assume the explicit barrier exists
# and cannot prove synchronization after its removal.
mock_path = DATA / "bench_records/v736_v738/audit_v736_v738_cpu.py"
mock_tree = ast.parse(mock_path.read_text(encoding="utf-8"))
mock_nodes = [node for node in mock_tree.body
              if isinstance(node, (ast.ClassDef, ast.FunctionDef))
              and node.name in ("Tensor", "run_mock")]
assert len(mock_nodes) == 2
namespace = {"ast": ast, "sys": sys, "types": types, "TREES": TREES, "FILES": FILES}
exec(compile(ast.Module(body=mock_nodes, type_ignores=[]), str(mock_path), "exec"), namespace)
mock = namespace["run_mock"]
shapes = ((1, 512, 256), (8, 512, 256), (16, 2048, 8192), (32, 7168, 2048),
          (32, 4096, 2048), (32, 7168, 1024), (64, 7168, 2048))
for parent, candidate in ((739, 741), (740, 742)):
    for experts, hidden, intermediate in shapes:
        for dtype in ("float16", "float32"):
            expected = mock(parent, experts, hidden, intermediate, dtype)
            actual = mock(candidate, experts, hidden, intermediate, dtype)
            if (experts, hidden, intermediate) == (32, 7168, 2048):
                expected[1] = (BUILDERS[candidate], *expected[1][1:])
            assert actual == expected
    for dtype in ("float16", "float32"):
        for padded, valid, blocks in ((0, 0, 0), (256, 0, 2), (256, 1, 2), (512, 129, 4)):
            expected = mock(parent, 32, 7168, 2048, dtype, padded, valid, blocks)
            actual = mock(candidate, 32, 7168, 2048, dtype, padded, valid, blocks)
            expected[1] = (BUILDERS[candidate], *expected[1][1:])
            assert actual == expected
    print(f"v{candidate}: both route dtypes, target/fallback shapes, fresh inputs and two launches per call PASS")
    print(f"v{candidate} SHA256 {hashlib.sha256(FILES[candidate].read_bytes()).hexdigest()}")
print("CRITICAL: synchronization safety is NOT established. Inspect actual generated WAR/RAW barriers before GPU execution.")
print("Inherited route-load hoisting risk and zero-grid launch behavior are unchanged; neither is repaired here.")
