"""CPU-only exact-source/AST audit of the v729 end-K barrier relocation."""
import ast
import copy
import hashlib
from pathlib import Path

DATA = Path(__file__).resolve().parents[2]
BASE = DATA / "probe_v724_v720_e32_stage1_a_fragment_reuse.py"
CANDIDATE = DATA / "probe_v729_v724_e32_stage1_early_tail_barrier.py"
base_source = BASE.read_text(encoding="utf-8")
candidate_source = CANDIDATE.read_text(encoding="utf-8")
base_tree = ast.parse(base_source, filename=str(BASE))
candidate_tree = ast.parse(candidate_source, filename=str(CANDIDATE))
compile(candidate_tree, str(CANDIDATE), "exec")

before = (
    "                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)\n"
    "                    mma_emitter.mma(input_matrix3, weight_matrix, up_local)\n"
    "                    T.sync_threads()"
)
after = (
    "                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)\n"
    "                    T.sync_threads()\n"
    "                    mma_emitter.mma(input_matrix3, weight_matrix, up_local)"
)
assert base_source.count(before) == 1
assert candidate_source.count(after) == 1
expected_source = base_source[base_source.index("import torch"):].replace(before, after)
assert candidate_source[candidate_source.index("import torch"):] == expected_source
print("PASS complete executable source differs only by moving the existing steady end-K barrier.")

def steady(tree):
    builder = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_moe_stage1_prefetch_giu_merge_a_reuse"
    )
    matches = [
        node for node in ast.walk(builder)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(k_steps - 1)"
    ]
    assert len(matches) == 1
    return matches[0]

expected_tree = copy.deepcopy(base_tree)
old_loop = steady(base_tree)
expected_loop = steady(expected_tree)
candidate_loop = steady(candidate_tree)
assert [ast.unparse(node) for node in old_loop.body[-3:]] == [
    "mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)",
    "mma_emitter.mma(input_matrix3, weight_matrix, up_local)",
    "T.sync_threads()",
]
expected_loop.body[-2], expected_loop.body[-1] = expected_loop.body[-1], expected_loop.body[-2]
assert ast.dump(expected_tree) == ast.dump(candidate_tree)
assert [ast.unparse(node) for node in candidate_loop.body[-3:]] == [
    "mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)",
    "T.sync_threads()",
    "mma_emitter.mma(input_matrix3, weight_matrix, up_local)",
]
print("PASS whole-module AST equals one adjacent statement swap; terminal, dispatch, caches, passes and every other path unchanged.")

def calls_without_sync(loop):
    return [
        ast.unparse(node) for node in loop.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "T.sync_threads"
        )
    ]

def sync_count(loop):
    return sum(
        isinstance(node, ast.Call) and ast.unparse(node.func) == "T.sync_threads"
        for node in ast.walk(loop)
    )

assert calls_without_sync(old_loop) == calls_without_sync(candidate_loop)
assert sync_count(old_loop) == sync_count(candidate_loop) == 2
assert ast.unparse(candidate_loop.body[-1]) == "mma_emitter.mma(input_matrix3, weight_matrix, up_local)"
print("PASS all steady copies/fragment loads/MMA retain exact order; 2 explicit steady barriers remain; only a register-operand MMA follows the moved barrier.")
print("v729 SHA256", hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())
print("Static/CPU audit only: generated-source barrier placement, GPU correctness and performance are not implied.")
