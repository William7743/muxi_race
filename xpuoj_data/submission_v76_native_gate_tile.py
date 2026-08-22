"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM
(v76: native synchronous Gate M128xN128xK64 tile)

v19 (swizzle=16) case3 13.03ms 反而变慢（权重 L2 命中率下降）→ swizzle=4 保留。
v20：跳过 pure-padding block（actual_rows==0）的整个 GEMM 循环。
case3 的 padded_total=11136（87 blocks），有效 9088 → 2048 padding 行（16 blocks），
其中 count 恰为 128 倍数的 expert 产生纯 padding block，白跑 112 ki GEMM。
G_S/U_S 的 padding block 不写 ws；D_S 的 padding block 显式写 out=0。
if 包住单个 Pipelined 循环（v14c 的 pipeline 报错是 merged 的 ws 双写，与此不同）。

仅把 Gate 的稳定 M128xN128xK64 tile 改为 import_source 原生同步微内核：
128-bit global→LDS、显式 XOR swizzle、原生 float16x4 MFMA。Up/Down 完全保持
稳定版，用于首先验证完整外部 tile 的正确性和单阶段代码质量。
"""
import torch
import tilelang
import tilelang.language as T


_EXTERN_SOURCE = r"""
#include <tl_templates/maca/common.h>

TL_DEVICE int moe_swizzle64(int row, int col) {
    return row * 64 + ((((col >> 3) ^ (row & 7)) << 3) + (col & 7));
}

TL_DEVICE void moe_gate_m128n128k64(
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
    const int warp_m = wave & 1;
    const int warp_n = wave >> 1;

    float32x4 accum[4][4] = {};

    for (int ko = 0; ko < hidden; ko += 64) {
        #pragma unroll
        for (int q = tid; q < 1024; q += 256) {
            const int row = q >> 3;
            const int group = q & 7;
            const int phys = row * 64 + ((group ^ (row & 7)) << 3);
            *reinterpret_cast<float16x8*>(sa + phys) =
                *reinterpret_cast<const float16x8*>(
                    x + (block_start + row) * hidden + ko + group * 8);
            *reinterpret_cast<float16x8*>(sb + phys) =
                *reinterpret_cast<const float16x8*>(
                    w + ((expert * intermediate + block_n * 128 + row) * hidden)
                        + ko + group * 8);
        }
        __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            float16x4 af[4];
            float16x4 bf[4];
            const int frag_col = ki * 16 + (lane >> 4) * 4;
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const int row = warp_m * 64 + wi * 16 + (lane & 15);
                af[wi] = *reinterpret_cast<const float16x4*>(
                    sa + moe_swizzle64(row, frag_col));
            }
            #pragma unroll
            for (int wj = 0; wj < 4; ++wj) {
                const int row = warp_n * 64 + wj * 16 + (lane & 15);
                bf[wj] = *reinterpret_cast<const float16x4*>(
                    sb + moe_swizzle64(row, frag_col));
            }
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                #pragma unroll
                for (int wj = 0; wj < 4; ++wj) {
                    accum[wi][wj] = __builtin_mxc_mma_16x16x16f16(
                        bf[wj], af[wi], accum[wi][wj]);
                }
            }
        }
        __syncthreads();
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
            T.import_source(_EXTERN_SOURCE)
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if actual_rows > 0:
                T.call_extern(
                    "moe_gate_m128n128k64",
                    T.access_ptr(stacked_expert_tokens, "r"),
                    T.access_ptr(gate_w, "r"),
                    T.access_ptr(ws, "w"),
                    T.access_ptr(xs, "rw"),
                    T.access_ptr(wts, "rw"),
                    hidden,
                    intermediate,
                    expert_id,
                    block_start,
                    by,
                    actual_rows,
                    dtype=dtype,
                )

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
