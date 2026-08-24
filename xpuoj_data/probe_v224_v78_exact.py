"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM
(v78: native merged Gate M256/N128/K32 register pipeline)

v66 的正确 M256/N128/K64 @512 合并需要 48KB shared，只能单 block 驻留，得分 74。
本版只改变 Gate 合并核：用 import_source 原生同步微内核实现 K32、512线程，
shared 24KB。每个线程维持与稳定 M128 tile 相同的64个FP32累加元素，同时把权重
跨两个128-row块复用。本版进一步用普通同步128-bit global load预取下一K32到
线程寄存器，在当前tile的MFMA期间隐藏global latency；不使用禁用的async/bsm。
未合并 Gate、Up、Down 保持稳定 T.gemm 路径。

v19 (swizzle=16) case3 13.03ms 反而变慢（权重 L2 命中率下降）→ swizzle=4 保留。
v20：跳过 pure-padding block（actual_rows==0）的整个 GEMM 循环。
case3 的 padded_total=11136（87 blocks），有效 9088 → 2048 padding 行（16 blocks），
其中 count 恰为 128 倍数的 expert 产生纯 padding block，白跑 112 ki GEMM。
G_S/U_S 的 padding block 不写 ws；D_S 的 padding block 显式写 out=0。
if 包住单个 Pipelined 循环（v14c 的 pipeline 报错是 merged 的 ws 双写，与此不同）。
"""
import torch
import tilelang
import tilelang.language as T


_EXTERN_SOURCE = r"""
#include <tl_templates/maca/common.h>

TL_DEVICE int moe_swizzle32(int row, int col) {
    return row * 32 + ((((col >> 3) ^ (row & 3)) << 3) + (col & 7));
}

TL_DEVICE void moe_gate_m256n128k32(
    const half_t* x,
    const half_t* w,
    half_t* y,
    half_t* sa,
    half_t* sb,
    int hidden,
    int intermediate,
    int expert,
    int block_start,
    int block_n,
    int actual_rows) {
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int warp_m = wave & 3;
    const int warp_n = wave >> 2;
    float32x4 accum[4][4] = {};

    const int a_row0 = tid >> 2;
    const int a_row1 = a_row0 + 128;
    const int group = tid & 3;
    const int a_phys0 = a_row0 * 32 + ((group ^ (a_row0 & 3)) << 3);
    const int a_phys1 = a_row1 * 32 + ((group ^ (a_row1 & 3)) << 3);
    const int b_row = tid >> 2;
    const int b_phys = b_row * 32 + ((group ^ (b_row & 3)) << 3);
    float16x8 ga0 = *reinterpret_cast<const float16x8*>(
        x + (block_start + a_row0) * hidden + group * 8);
    float16x8 ga1 = *reinterpret_cast<const float16x8*>(
        x + (block_start + a_row1) * hidden + group * 8);
    float16x8 gb = *reinterpret_cast<const float16x8*>(
        w + ((expert * intermediate + block_n * 128 + b_row) * hidden)
            + group * 8);
    asm(";--------------");
    *reinterpret_cast<float16x8*>(sa + a_phys0) = ga0;
    *reinterpret_cast<float16x8*>(sa + a_phys1) = ga1;
    *reinterpret_cast<float16x8*>(sb + b_phys) = gb;
    asm(";--------------");
    __syncthreads();

    for (int ko = 0; ko < hidden; ko += 32) {
        if (ko + 32 < hidden) {
            ga0 = *reinterpret_cast<const float16x8*>(
                x + (block_start + a_row0) * hidden + ko + 32 + group * 8);
            ga1 = *reinterpret_cast<const float16x8*>(
                x + (block_start + a_row1) * hidden + ko + 32 + group * 8);
            gb = *reinterpret_cast<const float16x8*>(
                w + ((expert * intermediate + block_n * 128 + b_row) * hidden)
                    + ko + 32 + group * 8);
        }
        float16x4 af[2][4];
        float16x4 bf[2][4];
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            const int slot = ki;
            const int frag_col = ki * 16 + (lane >> 4) * 4;
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const int row = warp_m * 64 + wi * 16 + (lane & 15);
                af[slot][wi] = *reinterpret_cast<const float16x4*>(
                    sa + moe_swizzle32(row, frag_col));
            }
            #pragma unroll
            for (int wj = 0; wj < 4; ++wj) {
                const int row = warp_n * 64 + wj * 16 + (lane & 15);
                bf[slot][wj] = *reinterpret_cast<const float16x4*>(
                    sb + moe_swizzle32(row, frag_col));
            }
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                #pragma unroll
                for (int wj = 0; wj < 4; ++wj) {
                    accum[wi][wj] = __builtin_mxc_mma_16x16x16f16(
                        bf[slot][wj], af[slot][wi], accum[wi][wj]);
                }
            }
        }
        __syncthreads();
        if (ko + 32 < hidden) {
            asm(";--------------");
            *reinterpret_cast<float16x8*>(sa + a_phys0) = ga0;
            *reinterpret_cast<float16x8*>(sa + a_phys1) = ga1;
            *reinterpret_cast<float16x8*>(sb + b_phys) = gb;
            asm(";--------------");
            __syncthreads();
        }
    }

    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        #pragma unroll
        for (int wj = 0; wj < 4; ++wj) {
            const int row = warp_m * 64 + wi * 16 + (lane & 15);
            const int col0 = warp_n * 64 + wj * 16 + (lane >> 4) * 4;
            const float* cv = reinterpret_cast<const float*>(&accum[wi][wj]);
            if (row < actual_rows) {
                #pragma unroll
                for (int lid = 0; lid < 4; ++lid) {
                    y[(block_start + row) * intermediate
                      + block_n * 128 + col0 + lid] = half_t(cv[lid]);
                }
            }
        }
    }
}
"""


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
    gu_k_pack = 2 if hidden >= 7000 else 1
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
        # ---- G_M: merge adjacent same-expert blocks into one M256 GEMM ----
        with T.Kernel(num_pairs, T.ceildiv(intermediate, block_n1), threads=512) as (bx, by):
            T.import_source(_EXTERN_SOURCE)
            xs_m = T.alloc_shared((256, merge_k), dtype=dtype)
            wts_m = T.alloc_shared((block_n1, merge_k), dtype=dtype)

            b0 = bx * 2
            b1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(b1 > b0, 1, 0)
            same = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[b1], 1, 0)
            active = has1 * same
            expert_id = group_idx_for_bx[b0]
            block_start = b0 * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(256, group_size - token_offset))

            if active == 1:
                T.call_extern(
                    "moe_gate_m256n128k32",
                    T.access_ptr(stacked_expert_tokens, "r"),
                    T.access_ptr(gate_w, "r"),
                    T.access_ptr(ws, "w"),
                    T.access_ptr(xs_m, "rw"),
                    T.access_ptr(wts_m, "rw"),
                    hidden,
                    intermediate,
                    expert_id,
                    block_start,
                    by,
                    actual_rows,
                    dtype=dtype,
                )

        # ---- G_S: gate GEMM, blocks not covered by G_M ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

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
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset)) * (1 - covered)

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
                    T.gemm(xs, wts, acc, transpose_B=True, k_pack=gu_k_pack)

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
                    T.gemm(xs, wts, acc, transpose_B=True, k_pack=gu_k_pack)

                for i, j in T.Parallel(block_token, block_n1):
                    if i < actual_rows:
                        ws[block_start + i, by * block_n1 + j] = (
                            ws[block_start + i, by * block_n1 + j]
                            * (1.0 / (1.0 + T.exp2(-ws[block_start + i, by * block_n1 + j] * scale)))
                            * acc[i, j]
                        )

        # ---- D_S: down GEMM, single 128-row blocks ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, block_n2), threads=threads_single) as (bx, by):
            hs = T.alloc_shared((block_token, block_k2), dtype=dtype)
            ds = T.alloc_shared((block_n2, block_k2), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n2), dtype=accum_dtype)

            T.use_swizzle(swizzle_panel)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

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
                            acc[i, j] * routed_expert_weights[raw_start + token_offset + i]
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
