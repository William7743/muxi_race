"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM
(v84: correct Down-only M256xN128xK32 merge)

v19 (swizzle=16) case3 13.03ms 反而变慢（权重 L2 命中率下降）→ swizzle=4 保留。
v20：跳过 pure-padding block（actual_rows==0）的整个 GEMM 循环。
case3 的 padded_total=11136（87 blocks），有效 9088 → 2048 padding 行（16 blocks），
其中 count 恰为 128 倍数的 expert 产生纯 padding block，白跑 112 ki GEMM。
G_S/U_S 的 padding block 不写 ws；D_S 的 padding block 显式写 out=0。
if 包住单个 Pipelined 循环（v14c 的 pipeline 报错是 merged 的 ws 双写，与此不同）。

Gate/Up 保持75分稳定路径；仅 Down 对相邻同专家的两个128-row block做M256合并，
复用Down权重。相对v83将合并核K64改为K32，shared从约49KB降至约25KB，目标是
恢复双block驻留；仍用512线程把accumulator保持在64 FP32/线程。
"""
import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    block_token=128,
    block_n1=128,
    block_k1=64,
    block_n2=128,
    block_k2=64,
    threads_single=256,
    num_stages=1,
    swizzle_panel=4,
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    output_shape = (total_padded_tokens, hidden)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)
    weights_shape = (total_valid_tokens,)
    num_pairs = (num_blocks_m + 1) // 2
    merge_k = 32

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor(weights_shape, T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        ws: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- G_S: gate GEMM, single 128-row blocks ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if actual_rows > 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + block_token,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        xs,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * block_n1 : (by + 1) * block_n1,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        wts,
                    )
                    T.gemm(xs, wts, acc, transpose_B=True)

                for i, j in T.Parallel(block_token, block_n1):
                    if i < actual_rows:
                        ws[block_start + i, by * block_n1 + j] = acc[i, j]

        # ---- U_S: up GEMM + 就地 silu, single 128-row blocks ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if actual_rows > 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + block_token,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        xs,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * block_n1 : (by + 1) * block_n1,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        wts,
                    )
                    T.gemm(xs, wts, acc, transpose_B=True)

                for i, j in T.Parallel(block_token, block_n1):
                    if i < actual_rows:
                        ws[block_start + i, by * block_n1 + j] = (
                            ws[block_start + i, by * block_n1 + j]
                            * (1.0 / (1.0 + T.exp2(-ws[block_start + i, by * block_n1 + j] * scale)))
                            * acc[i, j]
                        )

        # ---- D_M: merge adjacent same-expert blocks for Down ----
        with T.Kernel(num_pairs, T.ceildiv(hidden, block_n2), threads=512) as (bx, by):
            hs_m = T.alloc_shared((256, merge_k), dtype=dtype)
            ds_m = T.alloc_shared((block_n2, merge_k), dtype=dtype)
            rwv = T.alloc_shared((256,), dtype=T.float32)
            acc_m = T.alloc_fragment((256, block_n2), dtype=accum_dtype)

            T.use_swizzle(swizzle_panel)

            b0 = bx * 2
            b1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(b1 > b0, 1, 0)
            same = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[b1], 1, 0)
            active = has1 * same
            expert_id = group_idx_for_bx[b0]
            block_start = b0 * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(256, group_size - token_offset))
            rw_last = total_valid_tokens - 1

            if active == 1:
                for i in T.Parallel(256):
                    rwv[i] = T.if_then_else(
                        i < actual_rows,
                        routed_expert_weights[
                            T.min(raw_start + token_offset + i, rw_last)
                        ],
                        0.0,
                    )

                T.clear(acc_m)
                for k in T.Pipelined(T.ceildiv(intermediate, merge_k), num_stages=num_stages):
                    T.copy(
                        ws[
                            block_start : block_start + 256,
                            k * merge_k : (k + 1) * merge_k,
                        ],
                        hs_m,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * block_n2 : (by + 1) * block_n2,
                            k * merge_k : (k + 1) * merge_k,
                        ],
                        ds_m,
                    )
                    T.gemm(hs_m, ds_m, acc_m, transpose_B=True)

                for i, j in T.Parallel(256, block_n2):
                    out[block_start + i, by * block_n2 + j] = T.if_then_else(
                        i < actual_rows,
                        acc_m[i, j] * rwv[i],
                        0.0,
                    )

        # ---- D_S: down GEMM, blocks not covered by D_M ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, block_n2), threads=threads_single) as (bx, by):
            hs = T.alloc_shared((block_token, block_k2), dtype=dtype)
            ds = T.alloc_shared((block_n2, block_k2), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n2), dtype=accum_dtype)

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            next_b = T.min(bx + 1, num_blocks_m - 1)
            has_next = T.if_then_else(next_b > bx, 1, 0)
            same_next = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[next_b], 1, 0)
            prev_b = T.max(bx - 1, 0)
            same_prev = T.if_then_else(group_idx_for_bx[prev_b] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has_next * same_next, same_prev)
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
                if actual_rows > 0:
                    T.clear(acc)
                    for k in T.Pipelined(T.ceildiv(intermediate, block_k2), num_stages=num_stages):
                        T.copy(
                            ws[
                                block_start : block_start + block_token,
                                k * block_k2 : (k + 1) * block_k2,
                            ],
                            hs,
                        )
                        T.copy(
                            down_w[
                                expert_id,
                                by * block_n2 : (by + 1) * block_n2,
                                k * block_k2 : (k + 1) * block_k2,
                            ],
                            ds,
                        )
                        T.gemm(hs, ds, acc, transpose_B=True)

                    for i, j in T.Parallel(block_token, block_n2):
                        if i < actual_rows:
                            out[block_start + i, by * block_n2 + j] = (
                                acc[i, j]
                                * routed_expert_weights[raw_start + token_offset + i]
                            )
                        else:
                            out[block_start + i, by * block_n2 + j] = 0
                else:
                    for i, j in T.Parallel(block_token, block_n2):
                        out[block_start + i, by * block_n2 + j] = 0

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _moe_forward_kernel(*key)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _get_workspace(stacked_expert_tokens, intermediate):
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
        str(stacked_expert_tokens.dtype),
    )
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = torch.empty(
            (int(stacked_expert_tokens.shape[0]), int(intermediate)),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = ws
    return ws


def run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
):
    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])

    ws = _get_workspace(stacked_expert_tokens, intermediate)
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
    )
    kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        ws,
        out,
    )
