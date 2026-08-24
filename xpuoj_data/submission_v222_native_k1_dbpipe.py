# XPU-OJ v222: v138 with kernel1 replaced by native MFMA double-buffer pipeline
#
# 探针目标：kernel1（占 59% 时间）改用 import_source 原生 MACA C++ 实现
# M128xN128 融合 Gate/Up 双累加器 @256 线程，与 v138 T.gemm 路径资源画像一致：
#   1. A / gate_w / up_w 三组 shared 各双缓冲（共 48KB，占用率与 v138 相同 1 block/SM）
#   2. 每 K32 步只 1 次 barrier（v138 串行覆盖路径需 2 次显式 sync，且 T.Pipelined
#      无法安全流水同一权重 buffer 覆盖——v145 已证）
#   3. global→register 预取在 frag 读 + MMA 之前发射（同步软件流水，v78 已证合法有效）
#   4. A-fragment 每步只从 LDS 读一次、gate/up 两套 MMA 复用（A 侧 LDS 流量减半）
# fragment ABI、swizzle32、epilogue 公式严格照抄 v78/v138 已验证实现。
# hidden/intermediate 不满足 32/128 整除时编译期回退原 T.gemm 路径。
import torch
import tilelang
import tilelang.language as T


_EXTERN_SOURCE = r"""
#include <tl_templates/maca/common.h>

TL_DEVICE int moe_swizzle32(int row, int col) {
    return row * 32 + ((((col >> 3) ^ (row & 3)) << 3) + (col & 7));
}

// Fused Gate+Up GEMM M128xN128 per K32 step, 256 threads (4 warps x 64 lanes).
// sa/sg/su: double buffers, slot stride 4096 halves (128 rows x 32 cols each).
// Pipeline: prologue writes slot0; loop k prefetches k+1 into registers,
// reads fragments of slot k%2, runs gate+up MMAs reusing A frags once,
// stores registers into slot (k+1)%2, then ONE __syncthreads().
TL_DEVICE void moe_fused_gu_m128n128k32_db(
    const half_t* x,
    const half_t* gw,
    const half_t* uw,
    half_t* y,
    half_t* sa,
    half_t* sg,
    half_t* su,
    int hidden,
    int intermediate,
    int expert,
    int block_start,
    int block_n0,
    int actual_rows) {
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int wave = tid >> 6;
    const int warp_m = wave & 1;
    const int warp_n = wave >> 1;
    const int group = tid & 3;
    float32x4 accg[4][4] = {};
    float32x4 accu[4][4] = {};

    // This thread stages token rows a_row0/a_row1 and weight rows w_row0/w_row1.
    const int a_row0 = tid >> 2;
    const int a_row1 = a_row0 + 64;
    const int p0 = moe_swizzle32(a_row0, group * 8);
    const int p1 = moe_swizzle32(a_row1, group * 8);

    const long xrow0 = (long)(block_start + a_row0);
    const long xrow1 = (long)(block_start + a_row1);
    const half_t* xp0 = x + xrow0 * hidden + group * 8;
    const half_t* xp1 = x + xrow1 * hidden + group * 8;
    const long wbase = (long)expert * intermediate + block_n0;
    const half_t* gp0 = gw + (wbase + a_row0) * hidden + group * 8;
    const half_t* gp1 = gw + (wbase + a_row1) * hidden + group * 8;
    const half_t* up0 = uw + (wbase + a_row0) * hidden + group * 8;
    const half_t* up1 = uw + (wbase + a_row1) * hidden + group * 8;

    float16x8 xa0 = *(const float16x8*)(xp0);
    float16x8 xa1 = *(const float16x8*)(xp1);
    float16x8 ga0 = *(const float16x8*)(gp0);
    float16x8 ga1 = *(const float16x8*)(gp1);
    float16x8 ua0 = *(const float16x8*)(up0);
    float16x8 ua1 = *(const float16x8*)(up1);

    asm(";----prologue store----");
    *(float16x8*)(sa + 0 * 4096 + p0) = xa0;
    *(float16x8*)(sa + 0 * 4096 + p1) = xa1;
    *(float16x8*)(sg + 0 * 4096 + p0) = ga0;
    *(float16x8*)(sg + 0 * 4096 + p1) = ga1;
    *(float16x8*)(su + 0 * 4096 + p0) = ua0;
    *(float16x8*)(su + 0 * 4096 + p1) = ua1;
    asm(";----prologue done----");
    __syncthreads();

    for (int ko = 0; ko < hidden; ko += 32) {
        const int cur = (ko >> 5) & 1;
        if (ko + 32 < hidden) {
            xa0 = *(const float16x8*)(xp0 + ko + 32);
            xa1 = *(const float16x8*)(xp1 + ko + 32);
            ga0 = *(const float16x8*)(gp0 + ko + 32);
            ga1 = *(const float16x8*)(gp1 + ko + 32);
            ua0 = *(const float16x8*)(up0 + ko + 32);
            ua1 = *(const float16x8*)(up1 + ko + 32);
        }
        half_t* sac = sa + cur * 4096;
        half_t* sgc = sg + cur * 4096;
        half_t* suc = su + cur * 4096;

        float16x4 af[2][4];
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            const int frag_col = ki * 16 + (lane >> 4) * 4;
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const int row = warp_m * 64 + wi * 16 + (lane & 15);
                af[ki][wi] = *(const float16x4*)(sac + moe_swizzle32(row, frag_col));
            }
        }
        #pragma unroll
        for (int ki = 0; ki < 2; ++ki) {
            const int frag_col = ki * 16 + (lane >> 4) * 4;
            #pragma unroll
            for (int wj = 0; wj < 4; ++wj) {
                const int row = warp_n * 64 + wj * 16 + (lane & 15);
                const float16x4 bg = *(const float16x4*)(sgc + moe_swizzle32(row, frag_col));
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    accg[wi][wj] = __builtin_mxc_mma_16x16x16f16(bg, af[ki][wi], accg[wi][wj]);
                }
                const float16x4 bu = *(const float16x4*)(suc + moe_swizzle32(row, frag_col));
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    accu[wi][wj] = __builtin_mxc_mma_16x16x16f16(bu, af[ki][wi], accu[wi][wj]);
                }
            }
        }
        if (ko + 32 < hidden) {
            asm(";----next store----");
            *(float16x8*)(sa + (cur ^ 1) * 4096 + p0) = xa0;
            *(float16x8*)(sa + (cur ^ 1) * 4096 + p1) = xa1;
            *(float16x8*)(sg + (cur ^ 1) * 4096 + p0) = ga0;
            *(float16x8*)(sg + (cur ^ 1) * 4096 + p1) = ga1;
            *(float16x8*)(su + (cur ^ 1) * 4096 + p0) = ua0;
            *(float16x8*)(su + (cur ^ 1) * 4096 + p1) = ua1;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
        #pragma unroll
        for (int wj = 0; wj < 4; ++wj) {
            const int row = warp_m * 64 + wi * 16 + (lane & 15);
            const int col0 = warp_n * 64 + wj * 16 + (lane >> 4) * 4;
            if (row < actual_rows) {
                const float* cg = reinterpret_cast<const float*>(&accg[wi][wj]);
                const float* cu = reinterpret_cast<const float*>(&accu[wi][wj]);
                #pragma unroll
                for (int lid = 0; lid < 4; ++lid) {
                    const float g = cg[lid];
                    const float sig = 1.0f / (1.0f + exp2f(-g * 1.44269504f));
                    y[(block_start + row) * intermediate + block_n0 + col0 + lid]
                        = half_t(cu[lid] * (g * sig));
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
    bt1,
    bh1,
    be1,
    th1,
    bh2,
    be2,
    th2,
    weights_dtype,
):
    gu_k_pack = 2 if hidden >= 7000 else 1
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    native_k1 = (hidden % 32 == 0) and (intermediate % 128 == 0)

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Kernel 1: fused gate/up -> silu(gate)*up workspace ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            # swizzle(4, column)：v130/v138 实测最优块调度，两分支保持一致
            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))

            if native_k1:
                T.import_source(_EXTERN_SOURCE)
                sa_buf = T.alloc_shared((2, 128, 32), dtype=dtype)
                sg_buf = T.alloc_shared((2, 128, 32), dtype=dtype)
                su_buf = T.alloc_shared((2, 128, 32), dtype=dtype)
                if actual_rows > 0:
                    T.call_extern(
                        "moe_fused_gu_m128n128k32_db",
                        T.access_ptr(stacked_expert_tokens, "r"),
                        T.access_ptr(gate_w, "r"),
                        T.access_ptr(up_w, "r"),
                        T.access_ptr(up_logits, "w"),
                        T.access_ptr(sa_buf, "rw"),
                        T.access_ptr(sg_buf, "rw"),
                        T.access_ptr(su_buf, "rw"),
                        hidden,
                        intermediate,
                        expert_id,
                        block_start,
                        by * be1,
                        actual_rows,
                        dtype=dtype,
                    )
            else:
                input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
                weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
                gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
                up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

                active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)

                T.clear(gate_local)
                T.clear(up_local)

                for k in range(active_k_steps):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + bt1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        input_shared,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, k_pack=gu_k_pack)
                    T.sync_threads()
                    T.copy(
                        up_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, k_pack=gu_k_pack)
                    T.sync_threads()

                for i, j in T.Parallel(bt1, be1):
                    if i < actual_rows:
                        up_logits[block_start + i, by * be1 + j] = (
                            up_local[i, j]
                            * (
                                gate_local[i, j]
                                * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                            )
                        )

        # ---- Kernel 2: down GEMM × routed_weight -> out（padding 行写 0）----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, bh2), threads=th2) as (bx, by):
            up_shared = T.alloc_shared((bt1, be2), dtype=dtype)
            down_shared = T.alloc_shared((bh2, be2), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(bt1, group_size - token_offset))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)

            T.clear(out_local)

            for k in T.Pipelined(active_k_steps, num_stages=1):
                T.copy(
                    up_logits[
                        block_start : block_start + bt1,
                        k * be2 : (k + 1) * be2,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * bh2 : (by + 1) * bh2,
                        k * be2 : (k + 1) * be2,
                    ],
                    down_shared,
                )
                T.gemm(up_shared, down_shared, out_local, transpose_B=True)

            for i, j in T.Parallel(bt1, bh2):
                if i < actual_rows:
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
                else:
                    out[block_start + i, by * bh2 + j] = 0

    return kernel


def _pick_tiles(intermediate):
    return 128, 64, 128, 256


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    weights_dtype,
):
    bt1, bh1, be1, th1 = _pick_tiles(intermediate)
    bh2, be2, th2 = 128, 64, 256
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
        bt1,
        bh1,
        be1,
        th1,
        bh2,
        be2,
        th2,
        str(weights_dtype),
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
    if routed_expert_weights.dtype == torch.float32:
        weights_dtype = T.float32
    else:
        weights_dtype = T.float16
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
        weights_dtype,
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
