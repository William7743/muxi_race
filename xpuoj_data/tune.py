"""
调参脚本：对每个测试用例扫描 tile 配置，输出最优 kernel 时间。
用法: python tune.py --case 3 [--quick]
"""
import argparse
import itertools
import time

import torch

import local_harness as H
import submission as S

CASES = {
    1: dict(d_hidden=2048, d_expert=8192, n_routed_experts=16, group_sum=2272),
    2: dict(d_hidden=7168, d_expert=2048, n_routed_experts=32, group_sum=4544),
    3: dict(d_hidden=7168, d_expert=2048, n_routed_experts=64, group_sum=9088),
}


def bench_one(cfg, tensors, info, tile, warmup=5, iters=30):
    d_h = cfg["d_hidden"]
    d_e = cfg["d_expert"]
    n_exp = cfg["n_routed_experts"]
    mb = info["m_blocks"]
    raw = info["raw_total"]
    up = torch.empty((mb * 128, d_e), device="cuda", dtype=torch.float16)
    out = torch.empty((mb * 128, d_h), device="cuda", dtype=torch.float16)

    k = S._fused_moe_kernel.compile(
        d_hidden=d_h, d_expert=d_e, n_routed_experts=n_exp, m_blocks=mb, raw_total=raw,
        block_token=tile["block_token"], block_dhidden=tile["block_dhidden"],
        block_dexpert=tile["block_dexpert"], threads=tile["threads"],
        num_stages=tile["num_stages"],
    )
    args = (
        tensors["stacked_expert_tokens"], tensors["gate_w"], tensors["up_w"], tensors["down_w"],
        tensors["routed_expert_weights"], tensors["group_sizes"], tensors["group_offsets"],
        tensors["group_padded_offsets"], tensors["group_idx_for_bx"], up, out,
    )
    for _ in range(warmup):
        k(*args)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        k(*args)
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = CASES[args.case]
    tensors, info = H.gen_test_data(seed=81394, **cfg)
    print("case", args.case, cfg, "padded_total", info["padded_total"])

    if args.quick:
        tiles = [
            dict(block_token=128, block_dhidden=128, block_dexpert=128, threads=256, num_stages=s)
            for s in (1, 2, 3)
        ]
    else:
        tiles = []
        for bt, bd, be, th, ns in itertools.product(
            [128], [128, 256], [128, 256], [128, 256, 512], [1, 2, 3]
        ):
            tiles.append(dict(block_token=bt, block_dhidden=bd, block_dexpert=be, threads=th, num_stages=ns))

    results = []
    for t in tiles:
        try:
            ms = bench_one(cfg, tensors, info, t, warmup=3, iters=15)
            results.append((ms, t))
            print(f"{ms:8.3f} ms  {t}")
        except Exception as e:
            print(f"  FAIL {t}: {str(e)[:100]}")
        torch.cuda.empty_cache()

    results.sort()
    print("\n=== TOP 5 ===")
    for ms, t in results[:5]:
        print(f"{ms:8.3f} ms  {t}")


if __name__ == "__main__":
    main()
