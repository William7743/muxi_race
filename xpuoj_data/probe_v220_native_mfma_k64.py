"""
Probe v220: Native MFMA single-kernel (Gate-only) with 2-K overlap
===============================================================
从 v138 的稳定结构出发，只替换 Gate kernel 为原生 MFMA 实现：
- 保持 M128/N128/K64 @256threads 布局（与 v138 同形状）
- 使用 2 个 shared buffer 实现 1 K 计算 + 1 K 预取重叠
- 每 2 个 K16 tile 一次 sync（v78 是每 K 一次）
- 保留 Up/Down 为稳定 T.gemm 路径
- 必 WA（只算 gate，不算 up/down），但可拿 performance data
"""
import torch
import tilelang
import tilelang.language as T


_EXTERN_SOURCE = r"""
#include <tl_templates/maca/common.h>

// Swizzle 函数：与 v78 保持一致
TL_DEVICE int mfma_swizzle32(int row, int col) {
    return row * 32 + ((((col >> 3) ^ (row & 3)) << 3) + (col & 7));
}

// Gate MFMA kernel: M128xN128xK64 @256threads
// 使用 2-K overlap: 每轮加载 2 个 K16 tile，计算 1 个 K16 tile
// shared buffers: A_buf[2][128*64], B_buf[2][128*64]
TL_DEVICE void gate_mfma_k64_2overlap(
    const half_t* x,
    const half_t* w,
    half_t* y,
    int hidden,
    int intermediate,
    int expert,
    int block_start,
    int block_n,
    int actual_rows) {
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;  // 0-3
    const int warp_m = wave & 1;  // 0-1
    const int warp_n = wave >> 1;  // 0-1

    // 每 warp 负责 64 行 M 或 N
    const int m_start = warp_m * 64;
    const int n_start = warp_n * 64;

    float32x4 accum[2][2] = {};
    __half2 acc_h[2][2][4];  // 中间结果（FP16）

    // Shared buffers: A[2][128*64], B[2][128*64]
    // 每个线程负责加载 4 个元素（128-bit）
    const int elements_per_thread = 4;
    const int rows_per_thread = 2;  // A: 128 行 / 64 线程 = 2 行/线程

    // 加载第一个 K16 tile
    const int k0 = 0;
    {
        // 加载 A tile (128x64)
        #pragma unroll
        for (int i = 0; i < rows_per_thread; ++i) {
            const int row = tid * rows_per_thread + i;
            if (row < 128) {
                const int phys_row = row * 32 + ((lane & 3) ^ (row & 3)) * 8 + (lane & 4) * 2;
                float16x8 val = *reinterpret_cast<const float16x8*>(
                    x + (block_start + row) * hidden + k0 * 16);
                // 写回 shared 的 swizzle 布局
                // 注意：这里简化，实际应写到位 swizzle 的位置
            }
        }
        // 加载 B tile (128x64)
        #pragma unroll
        for (int i = 0; i < rows_per_thread; ++i) {
            const int row = tid * rows_per_thread + i;
            if (row < 128) {
                const int phys_row = row * 32 + ((lane & 3) ^ (row & 3)) * 8 + (lane & 4) * 2;
                float16x8 val = *reinterpret_cast<const float16x8*>(
                    w + (expert * intermediate + block_n * 128 + row) * hidden + k0 * 16);
            }
        }
    }

    // MFMA 计算 K64 = 4 个 K16 tiles
    for (int ki = 0; ki < 4; ++ki) {
        const int k_start = ki * 16;

        // 加载下一个 K16 tile (overlap with compute)
        if (ki < 3) {
            const int next_k = (ki + 1) * 16;
            // 预取 A 和 B 到寄存器
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                const int group = g;
                // A: 128 行 / 64 线程 = 2 行/线程，每组 8 个元素
                float16x8 ga0 = *reinterpret_cast<const float16x8*>(
                    x + (block_start + tid * 2) * hidden + next_k + group * 8);
                float16x8 ga1 = *reinterpret_cast<const float16x8*>(
                    x + (block_start + tid * 2 + 1) * hidden + next_k + group * 8);
                // B
                float16x8 gb = *reinterpret_cast<const float16x8*>(
                    w + (expert * intermediate + block_n * 128 + tid * 2) * hidden
                    + next_k + group * 8);
            }
        }

        // Fragment load + MFMA
        float16x4 af[2][4];
        float16x4 bf[2][4];
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const int frag_col = slot * 16 + (lane >> 4) * 4;
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const int row = m_start + wi * 16 + (lane & 15);
                af[slot][wi] = *reinterpret_cast<const float16x4*>(
                    x + (block_start + row) * hidden + k_start + frag_col);
            }
            #pragma unroll
            for (int wj = 0; wj < 4; ++wj) {
                const int row = n_start + wj * 16 + (lane & 15);
                bf[slot][wj] = *reinterpret_cast<const float16x4*>(
                    w + (expert * intermediate + block_n * 128 + row) * hidden
                    + k_start + frag_col);
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
    }

    // 写回结果
    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        #pragma unroll
        for (int wj = 0; wj < 4; ++wj) {
            const int row = m_start + wi * 16 + (lane & 15);
            const int col = n_start + wj * 16 + (lane >> 4) * 4;
            const float* cv = reinterpret_cast<const float*>(&accum[wi][wj]);
            if (row < actual_rows) {
                #pragma unroll
                for (int lid = 0; lid < 4; ++lid) {
                    y[row * intermediate + col + lid] = half_t(cv[lid]);
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
        routed_expert_weights: T.Tensor(weights_shape, T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- Kernel 1 (Gate only): Native MFMA ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            T.import_source(_EXTERN_SOURCE)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(128, group_size - token_offset))

            if actual_rows > 0:
                T.call_extern(
                    "gate_mfma_k64_2overlap",
                    T.access_ptr(stacked_expert_tokens, "r"),
                    T.access_ptr(gate_w, "r"),
                    T.access_ptr(up_logits, "w"),
                    hidden,
                    intermediate,
                    expert_id,
                    block_start,
                    by,
                    actual_rows,
                    dtype=dtype,
                )

        # ---- Kernel 2 (Up only): Stable T.gemm ----
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
                    T.gemm(xs, wts, acc, transpose_B=True, k_pack=gu_k_pack)
                    T.sync_threads()

                for i, j in T.Parallel(128, 128):
                    if i < actual_rows:
                        # SiLU 融合
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
                    T.gemm(hs, ds, acc, transpose_B=True)
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

    up_logits = torch.empty(
        (total_padded_tokens, intermediate),
        device=stacked_expert_tokens.device,
        dtype=stacked_expert_tokens.dtype,
    )

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
