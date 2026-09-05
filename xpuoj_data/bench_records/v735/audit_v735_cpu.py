"""CPU/source audit of v735's independent 2x2-warp GU32 interleaving.

Run: python xpuoj_data/bench_records/v735/audit_v735_cpu.py
No GPU access. Reuses the adjacent v731/v732 audit for unchanged ancestry and
host mocks; geometry below is separately derived for the new 2x2 layout.
This does not prove automatic compiler barriers, device correctness or speed.
"""

import ast
from collections import Counter
import copy
import hashlib
from pathlib import Path
import runpy


DATA = Path(__file__).resolve().parents[2]
FILES = {
    731: DATA / "probe_v731_v720_e32_stage1_concat_gu64_k64.py",
    735: DATA / "probe_v735_v731_e32_stage1_interleaved_gu32_warp2x2.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
NAME = "_moe_stage1_concat_gu_n128"
BUILDERS = {
    version: next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == NAME)
    for version, tree in TREES.items()
}


def expr(text):
    return ast.parse(text, mode="eval").body


def call_name(node):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return ast.unparse(node.value.func)
    return None


class ExpectedChanges(ast.NodeTransformer):
    """Exact whitelist applied only to a copy of the v731 target builder."""

    def visit_Call(self, node):
        if ast.unparse(node.func) == "TensorCoreIntrinEmitter":
            replacements = {"block_row_warps": 2, "block_col_warps": 2,
                            "warp_row_tiles": 64, "warp_col_tiles": 64}
            assert sum(kw.arg in replacements for kw in node.keywords) == 4
            for kw in node.keywords:
                if kw.arg in replacements:
                    kw.value = ast.Constant(replacements[kw.arg])
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if call_name(node) != "T.copy":
            return self.generic_visit(node)
        source = node.value.args[0]
        assert isinstance(source, ast.Subscript)
        if source.value.id == "stacked_expert_tokens":
            return node
        assert source.value.id in ("gate_w", "up_w")
        first, second = copy.deepcopy(node), copy.deepcopy(node)
        first.value.args[0].slice.elts[1] = ast.Slice(
            expr("by * output_tile"), expr("by * output_tile + 32"))
        second.value.args[0].slice.elts[1] = ast.Slice(
            expr("by * output_tile + 32"), expr("(by + 1) * output_tile"))
        regions = ((0, 32), (64, 96)) if source.value.id == "gate_w" else ((32, 64), (96, "be1"))
        for target, (start, stop) in zip((first, second), regions):
            target.value.args[1].slice.elts[0] = ast.Slice(expr(str(start)), expr(str(stop)))
        return [first, second]

    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "row_tile":
            assert ast.unparse(node.iter) == "T.serial(2)"
            node.iter = expr("T.serial(4)")
        elif isinstance(node.target, ast.Name) and node.target.id == "col_tile":
            assert ast.unparse(node.iter) == "T.serial(4)"
            node.target.id = "pair"
            node.iter = expr("T.serial(2)")
        return self.generic_visit(node)

    def visit_Assign(self, node):
        replacements = {
            "output_row": "warp_m * 64 + row_tile * 16 + row",
            "output_col": "warp_n * 32 + pair * 16 + col",
            "gate_slot": "row_tile * 16 + pair * 4 + local_id",
        }
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements:
            node.value = expr(replacements[node.targets[0].id])
            return node
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "gu_local":
            assert ast.unparse(node.slice) in ("gate_slot", "gate_slot + 16")
            if ast.unparse(node.slice) == "gate_slot + 16":
                node.slice = expr("gate_slot + 8")
        return self.generic_visit(node)


expected = copy.deepcopy(TREES[731])
position = next(i for i, node in enumerate(expected.body)
                if isinstance(node, ast.FunctionDef) and node.name == NAME)
expected.body[position] = ExpectedChanges().visit(expected.body[position])
assert ast.dump(expected) == ast.dump(TREES[735])
old_segment = ast.get_source_segment(SOURCES[731], BUILDERS[731])
new_segment = ast.get_source_segment(SOURCES[735], BUILDERS[735])
restored = SOURCES[735].replace(new_segment, old_segment)
assert restored[restored.index("import torch"):].rstrip() == SOURCES[731][SOURCES[731].index("import torch"):].rstrip()
print("Whole source/AST whitelist: emitter geometry, four B segments and paired epilogue only PASS")

new = BUILDERS[735]
values = {node.targets[0].id: node.value for node in ast.walk(new)
          if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
guard = next(node for node in ast.walk(new)
             if isinstance(node, ast.If) and ast.unparse(node.test) == "actual_rows > 0")
steady, terminal = guard.body[0].body, guard.body[2:]
assert [call_name(node) for node in steady] == ["T.copy"] * 5 + [None, "T.sync_threads"]
assert [call_name(node) for node in terminal] == ["T.copy"] * 5 + [None]
for body in (steady, terminal):
    assert [node.value.args[0].value.id for node in body[:5]] == [
        "gate_w", "gate_w", "stacked_expert_tokens", "up_w", "up_w"
    ]
    assert ast.unparse(body[5].iter) == "T.serial(bh1 // 16)"
    assert [ast.unparse(node) for node in body[5].body] == [
        "mma_emitter.ldmatrix_a(input_matrix, input_shared, ki)",
        "mma_emitter.ldmatrix_b(weight_matrix, weight_shared, ki)",
        "mma_emitter.mma(input_matrix, weight_matrix, gu_local)",
    ]


def evaluate(node, environment):
    return eval(compile(ast.Expression(node), "<v735-index>", "eval"),
                {"__builtins__": {}}, environment)


def bounds(node, env):
    assert isinstance(node, ast.Slice) and node.step is None
    return evaluate(node.lower, env), evaluate(node.upper, env)


# Reconstruct actual T.copy source/destination indices from the candidate AST.
# First/last N tiles, experts and steady/terminal K test all slice boundaries.
for by in (0, 31):
    for expert in (0, 31):
        for k in (0, 1, 110, 111):
            body = terminal if k == 111 else steady
            env = dict(by=by, expert_id=expert, k=k, terminal_k=k,
                       output_tile=64, bt1=128, be1=128, bh1=64, block_start=6016)
            shared = {}
            for statement in body[:5]:
                source, destination = statement.value.args[:2]
                kind = source.value.id
                source_coords = source.slice.elts
                n0, n1 = bounds(source_coords[-2], env)
                k0, k1 = bounds(source_coords[-1], env)
                assert (k0, k1) == (k * 64, (k + 1) * 64)
                if kind == "stacked_expert_tokens":
                    assert (n0, n1) == (6016, 6144)
                    assert isinstance(destination, ast.Name) and destination.id == "input_shared"
                    name, d0, d1 = "input_shared", 0, 128
                else:
                    assert evaluate(source_coords[0], env) == expert
                    assert n1 - n0 == 32
                    assert isinstance(destination, ast.Subscript)
                    name = destination.value.id
                    d0, d1 = bounds(destination.slice.elts[0], env)
                    assert bounds(destination.slice.elts[1], env) == (0, 64)
                    assert d1 - d0 == 32 and name == "weight_shared"
                for row in range(d0, d1):
                    for col in range(64):
                        key = name, row, col
                        assert key not in shared
                        shared[key] = (kind, n0 + row - d0, k0 + col)
            assert len(shared) == 2 * 128 * 64
            for row in range(128):
                for col in range(64):
                    kind = "gate_w" if (row // 32) % 2 == 0 else "up_w"
                    n = by * 64 + (row // 64) * 32 + row % 32
                    assert shared["weight_shared", row, col] == (kind, n, k * 64 + col)
                    assert shared["input_shared", row, col] == (
                        "stacked_expert_tokens", 6016 + row, k * 64 + col)
print("Current-input copy AST replay: G0/G1/A/U0/U1, all four B regions disjoint and complete PASS")


# Official default is_m_first=False:
# lane=tid%64, warp_m=(tid//64)%2, warp_n=(tid//128)%2.
# Official 64-lane C map is row=lane%16,col=local+(lane//16)*4.
def c_map(thread, row_tile, col_tile, local):
    lane, warp_m, warp_n = thread % 64, (thread // 64) % 2, (thread // 128) % 2
    return (warp_m * 64 + row_tile * 16 + lane % 16,
            warp_n * 64 + col_tile * 16 + local + (lane // 16) * 4)


def global_b(physical_column):
    return ("gate_w" if physical_column % 64 < 32 else "up_w",
            (physical_column // 64) * 32 + physical_column % 32)


all_c, output = Counter(), Counter()
for thread in range(256):
    for row_tile in range(4):
        for col_tile in range(4):
            for local in range(4):
                assert 0 <= row_tile * 16 + col_tile * 4 + local < 64
                all_c[c_map(thread, row_tile, col_tile, local)] += 1
        for pair in range(2):
            for local in range(4):
                lane = thread % 64
                env = dict(warp_m=(thread // 64) % 2, warp_n=(thread // 128) % 2,
                           row_tile=row_tile, pair=pair, local_id=local,
                           row=lane % 16, col=local + (lane // 16) * 4)
                row = evaluate(values["output_row"], env)
                col = evaluate(values["output_col"], env)
                gate_slot = evaluate(values["gate_slot"], env)
                assert gate_slot == row_tile * 16 + pair * 4 + local
                up_slot = gate_slot + 8
                assert up_slot == row_tile * 16 + (pair + 2) * 4 + local
                assert 0 <= gate_slot < 64 and 0 <= up_slot < 64
                gate_coordinate = c_map(thread, row_tile, pair, local)
                up_coordinate = c_map(thread, row_tile, pair + 2, local)
                assert gate_coordinate[0] == up_coordinate[0] == row
                assert global_b(gate_coordinate[1]) == ("gate_w", col)
                assert global_b(up_coordinate[1]) == ("up_w", col)
                output[row, col] += 1
assert all_c == Counter({(r, c): 1 for r in range(128) for c in range(128)})
assert output == Counter({(r, c): 1 for r in range(128) for c in range(64)})
for rows in range(129):
    assert sum(n for (r, _), n in output.items() if r < rows) == rows * 64
assert {by * 64 + col for by in range(32) for col in range(64)} == set(range(2048))

# Logical fragment reads: each A/B element is read by two warps, with local
# 16-half operand arrays. These are not physical register or occupancy counts.
for operand in ("a", "b"):
    coverage = Counter()
    for thread in range(256):
        lane = thread % 64
        warp_axis = (thread // 64) % 2 if operand == "a" else (thread // 128) % 2
        for micro in range(4):
            for tile in range(4):
                for local in range(4):
                    assert 0 <= tile * 4 + local < 16
                    coverage[warp_axis * 64 + tile * 16 + lane % 16,
                             micro * 16 + (lane // 16) * 4 + local] += 1
    assert set(coverage) == {(r, c) for r in range(128) for c in range(64)}
    assert set(coverage.values()) == {2}
assert (128 + 128) * 64 * 2 == 32768
assert [k * 64 + micro * 16 for k in list(range(111)) + [111]
        for micro in range(4)] == list(range(0, 7168, 16))
print("2x2 geometry: 16384 C slots / 8192 Gate-Up +8 pairs, exact rows/K16/A-B coverage PASS")

# Full-module equality after the whitelist already proves the expression tree
# changes only its paired Up offset, with operation order and scale unchanged.
context = runpy.run_path(str(DATA / "bench_records/v731_v732/audit_v731_v732_cpu.py"))
context["FILES"].update(FILES)
context["TREES"].update(TREES)
mock = context["run_mock"]
for experts in (1, 8, 16, 32, 64):
    for dtype in ("float16", "float32"):
        shapes = (None, (4096, 2048), (7168, 4096)) if experts == 32 else (None,)
        for shape in shapes:
            assert mock(735, experts, dtype, shape) == mock(731, experts, dtype, shape)
        print(f"v735 E{experts:02} {dtype}: two fresh calls / two launches each, same dispatcher PASS")
print("v735 SHA256", hashlib.sha256(FILES[735].read_bytes()).hexdigest())
print("CPU audit PASS. Generated write/read barriers, actual numerical output and performance still need GPU checks.")
