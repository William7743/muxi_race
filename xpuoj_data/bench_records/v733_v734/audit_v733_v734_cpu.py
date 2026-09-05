"""Exact-source isolation audit for v733/v734; CPU only.

Run from any directory:
  python xpuoj_data/bench_records/v733_v734/audit_v733_v734_cpu.py

Also runs the existing adjacent v731/v732 geometry/source/mock audit. This is
not a GPU result and does not establish that the compiler removes all dynamic
indexing, or that unrolling improves performance.
"""

import ast
import copy
import hashlib
from pathlib import Path
import runpy


DATA = Path(__file__).resolve().parents[2]
FILES = {
    731: DATA / "probe_v731_v720_e32_stage1_concat_gu64_k64.py",
    732: DATA / "probe_v732_v720_e32_stage1_concat_gu64_k32.py",
    733: DATA / "probe_v733_v731_e32_stage1_concat_gu64_k64_local_unroll.py",
    734: DATA / "probe_v734_v732_e32_stage1_concat_gu64_k32_local_unroll.py",
}
SOURCES = {version: path.read_text(encoding="utf-8") for version, path in FILES.items()}
TREES = {version: ast.parse(source) for version, source in SOURCES.items()}
BUILDER = "_moe_stage1_concat_gu_n128"


def builder(tree):
    return next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == BUILDER)


def source_segment(version):
    return ast.get_source_segment(SOURCES[version], builder(TREES[version]))


for version, base, tile_k in ((733, 731, 64), (734, 732, 32)):
    base_segment, candidate_segment = source_segment(base), source_segment(version)
    assert base_segment.count("T.serial(") == 5
    assert candidate_segment == base_segment.replace("T.serial(", "T.unroll(")
    restored = SOURCES[version].replace(candidate_segment, base_segment)
    assert restored[restored.index("import torch"):].rstrip() == SOURCES[base][SOURCES[base].index("import torch"):].rstrip()

    # The full module, not only the new builder, must be identical after
    # restoring these five loop kinds. Decorators/imports/dispatch remain intact.
    normalized = copy.deepcopy(TREES[version])
    loops = sorted(
        [node for node in ast.walk(builder(normalized))
         if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
         and ast.unparse(node.iter.func) == "T.unroll"],
        key=lambda node: node.lineno,
    )
    assert [(ast.unparse(node.target), ast.unparse(node.iter))
            for node in loops] == [
        ("ki", "T.unroll(bh1 // 16)"),
        ("ki", "T.unroll(bh1 // 16)"),
        ("row_tile", "T.unroll(2)"),
        ("col_tile", "T.unroll(4)"),
        ("local_id", "T.unroll(4)"),
    ]
    for node in loops:
        assert not node.orelse
        node.iter.func.attr = "serial"
    assert ast.dump(normalized) == ast.dump(TREES[base])

    # No outer-K unroll or reordering: enumerate the same steady/terminal
    # reduction order and epilogue local slots for each explicitly unrolled loop.
    steps = 7168 // tile_k
    serial_order = [(k, micro) for k in range(steps) for micro in range(tile_k // 16)]
    expanded_order = []
    for k in list(range(steps - 1)) + [steps - 1]:
        expanded_order.extend((k, micro) for micro in tuple(range(tile_k // 16)))
    assert expanded_order == serial_order
    assert [k * tile_k + micro * 16 for k, micro in expanded_order] == list(range(0, 7168, 16))
    serial_slots = [(i * 32 + j * 4 + local, i * 32 + j * 4 + local + 16)
                    for i in range(2) for j in range(4) for local in range(4)]
    expanded_slots = []
    for i in (0, 1):
        for j in (0, 1, 2, 3):
            for local in (0, 1, 2, 3):
                expanded_slots.append((i * 32 + j * 4 + local, i * 32 + j * 4 + local + 16))
    assert expanded_slots == serial_slots
    assert len({slot for pair in expanded_slots for slot in pair}) == 64
    print(f"v{version}: complete source/AST only five loop kinds + header; exact K16/epilogue order PASS")

# Re-run the frozen design's complete CPU geometry/source audit and re-use its
# host-only mock for the new modules. No TileLang builders execute in this mock.
context = runpy.run_path(str(DATA / "bench_records/v731_v732/audit_v731_v732_cpu.py"))
context["FILES"].update(FILES)
context["TREES"].update(TREES)
run_mock = context["run_mock"]
for version, base in ((733, 731), (734, 732)):
    for experts in (1, 8, 16, 32, 64):
        for dtype in ("float16", "float32"):
            shapes = (None, (4096, 2048), (7168, 4096)) if experts == 32 else (None,)
            for shape in shapes:
                assert run_mock(version, experts, dtype, shape) == run_mock(base, experts, dtype, shape)
            print(f"v{version} E{experts:02} {dtype}: fresh-input forwarding/two launches each unchanged PASS")
    print(f"v{version} SHA256", hashlib.sha256(FILES[version].read_bytes()).hexdigest())
print("CPU isolation audit PASS; device codegen/numerical/performance validation remains separate.")
