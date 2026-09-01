#!/usr/bin/env python3
"""Minimal C500 TileLang smoke test matching the MoE tile geometry."""

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def build(threads):
    m = 128
    n = 128
    k = 64

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.float16),
        b: T.Tensor((n, k), T.float16),
        c: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            a_shared = T.alloc_shared((m, k), T.float16)
            b_shared = T.alloc_shared((n, k), T.float16)
            c_local = T.alloc_fragment((m, n), T.float32)
            T.clear(c_local)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            T.copy(c_local, c)

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def build_inplace(threads):
    m = 128
    n = 128
    k = 64

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.float16),
        b: T.Tensor((n, k), T.float16),
        c: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, threads=threads):
            a_shared = T.alloc_shared((m, k), T.float16)
            b_shared = T.alloc_shared((n, k), T.float16)
            c_local = T.alloc_fragment((m, n), T.float32)
            T.clear(c_local)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            T.copy(c_local, c)

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def build_grouped(threads):
    m = 128
    n = 128
    k = 64

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.float16),
        b: T.Tensor((1, n, k), T.float16),
        group_sizes: T.Tensor((1,), T.int32),
        group_padded_offsets: T.Tensor((2,), T.int32),
        group_idx: T.Tensor((1,), T.int32),
        c: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, 1, threads=threads) as (bx, by):
            a_shared = T.alloc_shared((m, k), T.float16)
            b_shared = T.alloc_shared((n, k), T.float16)
            c_local = T.alloc_fragment((m, n), T.float32)
            T.use_swizzle(4, order="column")
            expert = group_idx[bx]
            actual = T.max(0, T.min(m, group_sizes[expert] - (bx * m - group_padded_offsets[expert])))
            active = T.if_then_else(actual > 0, 1, 0)
            T.clear(c_local)
            for kk in range(active):
                T.copy(a[bx * m : (bx + 1) * m, kk * k : (kk + 1) * k], a_shared)
                T.copy(b[expert, by * n : (by + 1) * n, kk * k : (kk + 1) * k], b_shared)
                T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            for i, j in T.Parallel(m, n):
                if i < actual:
                    c[bx * m + i, by * n + j] = c_local[i, j]

    return kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    "tl.disable_safe_memory_legalize": True,
    "tl.disable_vectorize_256": True,
})
def build_grouped_dual(threads, k):
    m = 128
    n = 128
    tile_k = 64

    @T.prim_func
    def kernel(
        a: T.Tensor((m, k), T.float16),
        gate: T.Tensor((1, n, k), T.float16),
        up: T.Tensor((1, n, k), T.float16),
        group_sizes: T.Tensor((1,), T.int32),
        group_padded_offsets: T.Tensor((2,), T.int32),
        group_idx: T.Tensor((1,), T.int32),
        c: T.Tensor((m, n), T.float16),
    ):
        with T.Kernel(1, 1, threads=threads) as (bx, by):
            a_shared = T.alloc_shared((m, tile_k), T.float16)
            b_shared = T.alloc_shared((n, tile_k), T.float16)
            gate_local = T.alloc_fragment((m, n), T.float32)
            up_local = T.alloc_fragment((m, n), T.float32)
            T.use_swizzle(4, order="column")
            expert = group_idx[bx]
            actual = T.max(0, T.min(m, group_sizes[expert] - (bx * m - group_padded_offsets[expert])))
            active = T.if_then_else(actual > 0, k // tile_k, 0)
            T.clear(gate_local)
            T.clear(up_local)
            for kk in range(active):
                T.copy(a[bx * m : (bx + 1) * m, kk * tile_k : (kk + 1) * tile_k], a_shared)
                T.copy(
                    gate[expert, by * n : (by + 1) * n, kk * tile_k : (kk + 1) * tile_k],
                    b_shared,
                    coalesced_width=8,
                )
                T.gemm(a_shared, b_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
                T.sync_threads()
                T.copy(
                    up[expert, by * n : (by + 1) * n, kk * tile_k : (kk + 1) * tile_k],
                    b_shared,
                    coalesced_width=8,
                )
                T.gemm(a_shared, b_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
                T.sync_threads()
            for i, j in T.Parallel(m, n):
                if i < actual:
                    c[bx * m + i, by * n + j] = up_local[i, j] * (
                        gate_local[i, j]
                        * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * 1.44269504)))
                    )

    return kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--k", type=int, choices=(128, 2048), default=2048)
    args = parser.parse_args()
    if args.dual:
        kernel = build_grouped_dual(args.threads, args.k)
    elif args.grouped:
        kernel = build_grouped(args.threads)
    else:
        kernel = build_inplace(args.threads) if args.inplace else build(args.threads)
    width = args.k if args.dual else 64
    a = torch.randn((128, width), dtype=torch.float16).to("cuda")
    b = torch.randn((128, width), dtype=torch.float16).to("cuda")
    if args.dual:
        got = torch.empty((128, 128), dtype=torch.float16, device="cuda")
        up = torch.randn((1, 128, args.k), dtype=torch.float16).to("cuda")
        sizes = torch.tensor([127], dtype=torch.int32, device="cuda")
        offsets = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
        group_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        kernel(a, b.unsqueeze(0), up, sizes, offsets, group_idx, got)
    elif args.grouped:
        got = torch.empty((128, 128), dtype=torch.float16, device="cuda")
        sizes = torch.tensor([127], dtype=torch.int32, device="cuda")
        offsets = torch.tensor([0, 128], dtype=torch.int32, device="cuda")
        group_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
        kernel(a, b.unsqueeze(0), sizes, offsets, group_idx, got)
    elif args.inplace:
        got = torch.empty((128, 128), dtype=torch.float16, device="cuda")
        kernel(a, b, got)
    else:
        got = kernel(a, b)
    torch.cuda.synchronize()
    expected = a @ b.T
    if args.dual:
        expected = torch.nn.functional.silu(expected) * (a @ up[0].T)
    rows = 127 if (args.grouped or args.dual) else 128
    torch.testing.assert_close(got[:rows], expected[:rows], atol=0.05, rtol=0.05)
    print(f"threads={args.threads} OK max_abs={float((got[:rows] - expected[:rows]).abs().max())}")


if __name__ == "__main__":
    main()
