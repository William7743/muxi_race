"""
Probe v237: Native MFMA with 2x K overlap (based on v78)
======================================================
基于 v78 的已验证结构，尝试减少 sync 频率：
- 保持 M256/N128/K32 @512threads（与 v78 相同）
- 尝试每 2 个 K32 tile 一次 sync（v78 是每 K32 一次）
- 使用 2-K overlap: 预取下一个 K32 到寄存器
- 保留 Up/Down 为稳定 T.gemm 路径
- 必 WA（只算 gate），但可拿 performance data
"""
import torch
import tilelang
import tilelang.language as T


_EXTERN_SOURCE = r"""
#include <tl_templates/maca/common.h>

TL_DEVICE int moe_swizzle32(int row, int col) {
    return row * 32 + ((((col >> 3) ^ (row & 3)) << 3) + (col & 7));
}

// v78 结构的改进版：2-K32 overlap
// 每轮：加载 K[i+1] 到寄存器，计算 K[i] 的 MFMA
// 只在 K 边界 sync（每 2 个 K32 tile）
TL_DEVICE void gate_mfma_2k_overlap(
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

    // 预加载第一个 K32 tile
    float16x8 ga0 = *reinterpret_cast<const float16x8*>(
        x + (block_start + a_row0) * hidden + group * 8);
    float16x8 ga1 = *reinterpret_cast<const float16x8*>(
        x + (block_start + a_row1) * hidden + group * 8);
    float16x8 gb = *reinterpret_cast<const float16x8*>(
        w + ((expert * intermediate + block_n * 128 + b_row) * hidden)
            + group * 8);
    *reinterpret_cast<float16x8*>(sa + a_phys0) = ga0;
    *reinterpret_cast<float16x8*>(sa + a_phys1) = ga1;
    *reinterpret_cast<float16x8*>(sb + b_phys) = gb;
    __syncthreads();

    // K-loop: 每 2 个 K32 tile 一次 sync
    for (int ko = 0; ko < hidden; ko += 64) {
        // 加载下一 K64 tile 的 A 和 B 到寄存器
        float16x8 ga0_next = ga0;
        float16x8 ga1_next = ga1;
        float16x8 gb_next = gb;
        if (ko + 64 < hidden) {
            ga0_next = *reinterpret_cast<const float16x8*>(
                x + (block_start + a_row0) * hidden + ko + 64 + group * 8);
            ga1_next = *reinterpret_cast<const float16x8*>(
                x + (block_start + a_row1) * hidden + ko + 64 + group * 8);
            gb_next = *reinterpret_cast<const float16x8*>(
                w + ((expert * intermediate + block_n * 128 + b_row) * hidden)
                    + ko + 64 + group * 8);
        }

        // 计算当前 K64 (2 个 K32 tiles, 1 sync)
        float16x4 af[2][4];
        float16x4 bf[2][4];
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            const int k_start = ko + ki * 32;
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

        // 交换 buffer
        *reinterpret_cast<float16x8*>(sa + a_phys0) = ga0_next;
        *reinterpret_cast<float16x8*>(sa + a_phys1) = ga1_next;
        *reinterpret_cast<float16x8*>(sb + b_phys) = gb_next;
        ga0 = ga0_next;
        ga1 = ga1_next;
        gb = gb_next;
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


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    scale = 1.44269504
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
        routed_expert_weights: T.Tensor(weights_shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- Kernel 1 (Gate): Native MFMA with 2-K overlap ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=512) as (bx, by):
            T.import_source(_EXTERN_SOURCE)
            xs_m = T.alloc_shared((256, 32), dtype=dtype)
            wts_m = T.alloc_shared((128, 32), dtype=dtype)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(128, group_size - token_offset))

            if actual_rows > 0:
                T.call_extern(
                    "gate_mfma_2k_overlap",
                    T.access_ptr(stacked_expert_tokens, "r"),
                    T.access_ptr(gate_w, "r"),
                    T.access_ptr(up_logits, "w"),
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

        # ---- Kernel 2 (Up + SiLU): Stable T.gemm ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            xs = T.alloc_shared((128, 64), dtype=dtype)
            wts = T.alloc_shared((128, 64), dtype=dtype)
            acc = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4, order="column")
            gu_k_pack = 2 if hidden >= 7000 else 1

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(128, group_size - token_offset))

            if actual_rows > 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(hidden, 64), num_stages=1):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + 128,
                            k * 64 : (k + 1) * 64,
                        ],
                        xs,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * 128 : (by + 1) * 128,
                            k * 64 : (k + 1) * 64,
                        ],
                        wts,
                    )
                    T.gemm(xs, wts, acc, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()

                for i, j in T.Parallel(128, 128):
                    if i < actual_rows:
                        g = up_logits[block_start + i, by * 128 + j]
                        s = 1.0 / (1.0 + T.exp2(-g * scale))
                        up_logits[block_start + i, by * 128 + j] = acc[i, j] * g * s

        # ---- Kernel 3 (Down): Stable T.gemm ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, 128), threads=256) as (bx, by):
            hs = T.alloc_shared((128, 64), dtype=dtype)
            ds = T.alloc_shared((128, 64), dtype=dtype)
            acc = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(128, group_size - token_offset))

            if actual_rows > 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(intermediate, 64), num_stages=1):
                    T.copy(
                        up_logits[
                            block_start : block_start + 128,
                            k * 64 : (k + 1) * 64,
                        ],
                        hs,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * 128 : (by + 1) * 128,
                            k * 64 : (k + 1) * 64,
                        ],
                        ds,
                    )
                    T.gemm(hs, ds, acc, transpose_B=True, policy=T.GemmWarpPolicy.Square)
                    T.sync_threads()

                for i, j in T.Parallel(128, 128):
                    if i < actual_rows:
                        out[block_start + i, by * 128 + j] = (
                            acc[i, j] * routed_expert_weights[raw_start + token_offset + i]
                        )
                    else:
                        out[block_start + i, by * 128 + j] = 0
            else:
                for i, j in T.Parallel(128, 128):
                    out[block_start + i, by * 128 + j] = 0

    return kernel


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


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
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
    )
    up_logits = _WORKSPACE_CACHE.get(key)
    if up_logits is None:
        up_logits = torch.empty(
            (int(stacked_expert_tokens.shape[0]), int(intermediate)),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = up_logits
    return up_logits


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

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
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
        up_logits,
        out,
    )
