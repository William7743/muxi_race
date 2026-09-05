"""CPU-only audit of official MACA M32/N128/K64, 256-thread emitter geometry.

Reads the saved official mma_layout.py, utils.py and mma_macro_generator.py;
executes only named pure function/method ASTs, never imports torch/TileLang/TVM.
The default snapshot directory is the repository's parent. Override with
--snapshot-dir PATH. Missing snapshots fail; no guessed fallback maps are used.
This checks integer layout/coverage, not compilation, GPU correctness or speed.
"""

import argparse
import ast
from collections import Counter
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace


def extract(nodes, name):
    matches = [node for node in nodes
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def load_function(node, origin, env):
    assert not node.decorator_list, node.name
    print(f"AST source: {origin}:{node.lineno}-{node.end_lineno} {node.name}")
    clone = copy.deepcopy(node)
    # Type annotations are not layout arithmetic and can reference TVM classes.
    clone.returns = None
    for argument in clone.args.posonlyargs + clone.args.args + clone.args.kwonlyargs:
        argument.annotation = None
    exec(compile(ast.Module([clone], type_ignores=[]), str(origin), "exec"), env)
    return env[node.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path,
                        default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    snapshots = {}
    for name in ("mma_layout.py", "utils.py", "mma_macro_generator.py"):
        path = (args.snapshot_dir / name).resolve()
        if not path.is_file():
            parser.error(f"Missing official snapshot: {path}; use --snapshot-dir")
        data = path.read_bytes()
        snapshots[name] = (path, ast.parse(data.decode("utf-8")))
        print(f"Snapshot: {path} SHA256={hashlib.sha256(data).hexdigest()}")

    env = {}
    layout_path, layout_tree = snapshots["mma_layout.py"]
    for name in ("thread_id_shared_access_64x4_to_16x16_layout_A",
                 "shared_16x16_to_local_64x4_layout_A",
                 "thread_id_shared_access_64x4_to_16x16_layout_B",
                 "shared_16x16_to_local_64x4_layout_B",
                 "thread_id_shared_access_64x4_to_16x16_layout_C_n_m"):
        load_function(extract(layout_tree.body, name), layout_path, env)
    utils_path, utils_tree = snapshots["utils.py"]
    store_map = load_function(extract(utils_tree.body, "mma_store_index_map"),
                              utils_path, env)
    macro_path, macro_tree = snapshots["mma_macro_generator.py"]
    emitter = next(node for node in macro_tree.body
                   if isinstance(node, ast.ClassDef)
                   and node.name == "TensorCoreIntrinEmitter")
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in emitter.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"M_DIM", "N_DIM", "WARP_SIZE"}}
    assert constants == {"M_DIM": 16, "N_DIM": 16, "WARP_SIZE": 64}, constants
    binding = load_function(extract(emitter.body, "extract_thread_binding"), macro_path, env)
    load_maps = load_function(extract(emitter.body, "get_ldmatrix_index_map"), macro_path, env)
    store_layout = extract(emitter.body, "make_mma_store_layout")

    cfg = SimpleNamespace(WARP_SIZE=64, block_row_warps=2, block_col_warps=2,
                          is_m_first=False, k_dim=16, k_pack=1,
                          a_transposed=False, b_transposed=True)
    a_forward, a_reverse = load_maps(cfg, is_b=False)
    b_forward, b_reverse = load_maps(cfg, is_b=True)
    micro_inverse = {}
    for lane in range(64):
        for local in range(4):
            coordinate = tuple(store_map(lane, local))
            assert coordinate not in micro_inverse
            micro_inverse[coordinate] = (lane, local)
    assert set(micro_inverse) == {(r, c) for r in range(16) for c in range(16)}
    env.update(micro_size_x=16, micro_size_y=16, local_size_out=4,
               block_row_warps=2, block_col_warps=2, warp_rows=1, warp_cols=4,
               warp_size=64, is_m_first=False,
               inverse_mma_store_layout=SimpleNamespace(
                   map_indices=lambda indices: micro_inverse[tuple(indices)]))
    forward_thread = load_function(extract(store_layout.body, "forward_thread"), macro_path, env)
    forward_index = load_function(extract(store_layout.body, "forward_index"), macro_path, env)

    # Replay official store maps in both directions; all 256 threads own16 C values.
    output_owners = {}
    for row in range(32):
        for col in range(128):
            owner = (forward_thread(row, col), forward_index(row, col))
            assert 0 <= owner[0] < 256 and 0 <= owner[1] < 16
            assert owner not in output_owners
            output_owners[owner] = (row, col)
    assert set(output_owners) == {(t, slot) for t in range(256) for slot in range(16)}
    for thread in range(256):
        lane, warp_n, warp_m = binding(cfg, thread)
        assert (lane, warp_n, warp_m) == (thread % 64, thread // 128, thread // 64 % 2)
        for col_tile in range(4):
            for local in range(4):
                row, col = store_map(lane, local)
                coordinate = (warp_m * 16 + row, warp_n * 64 + col_tile * 16 + col)
                assert output_owners[(thread, col_tile * 4 + local)] == coordinate
    print("PASS C: 32x128 <-> 256 threads x16 slots, 4096-coordinate bijection")

    # The transposed B region is [N,K]; official get_ldmatrix_index_map chooses
    # its corresponding inverse map. Cross-warp duplicate operand reads are expected.
    a_counts, b_counts = Counter(), Counter()
    for thread in range(256):
        lane, warp_n, warp_m = binding(cfg, thread)
        for ki in range(4):
            for local in range(4):
                row, col = a_reverse(lane, local)
                assert a_forward(row, col) == (lane, local)
                coordinate = (warp_m * 16 + row, ki * 16 + col)
                assert 0 <= coordinate[0] < 32 and 0 <= coordinate[1] < 64
                a_counts[coordinate] += 1
            for col_tile in range(4):
                for local in range(4):
                    row, col = b_reverse(lane, local)
                    assert b_forward(row, col) == (lane, local)
                    coordinate = (warp_n * 64 + col_tile * 16 + row, ki * 16 + col)
                    assert 0 <= coordinate[0] < 128 and 0 <= coordinate[1] < 64
                    b_counts[coordinate] += 1
    assert set(a_counts) == {(r, k) for r in range(32) for k in range(64)}
    assert set(b_counts) == {(n, k) for n in range(128) for k in range(64)}
    assert set(a_counts.values()) == set(b_counts.values()) == {2}
    print("PASS A: 32x64 /2048 coordinates; B: 128x64 /8192 coordinates; each read twice")
    print("PASS logical local sizes: A4 half, each B16 half, C16 float (k_pack=1)")
    print("LIMIT: integer/pure-map audit only; no TileLang import, compilation, GPU run, "
          "numeric precision, performance or physical register/occupancy claim")


if __name__ == "__main__":
    main()
