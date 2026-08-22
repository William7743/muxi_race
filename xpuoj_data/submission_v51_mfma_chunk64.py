"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM  (v51: classless MFMA chunk64 barriers-half)

== 手工 MMA 合并版（tilelang.intrinsics 路线）==
目标：实现权重读取减半（合并相邻同专家 block 对）并绕开 T.gemm 的
"共享操作数双 gemm miscompile" 与 (256,128)@512 的性能惩罚。

依据 examples/maca/gemm/example_gemm_intrinsics.py（race 分支官方示例）：
- TensorCoreIntrinEmitter (MACA 16x16x16 MFMA, warp_size=64)
- merged tile: 256 行 x 128 列, 8 warps @512th (4x2 warp 网格, 每 warp 64x64)
- chunk=32, num_stages=1: shared = A(256,32)16K + B(128,32)8K = 24K -> 2 blocks/SM
- 手动 Parallel 装载 + T.annotate_layout swizzle (最后一维 32*16bit=512 可 swizzle)
- G_M 用 stmatrix 直写 global (pid_m/pid_n 重载)
- U_M/D_M 用自定义 fragment store 循环（复刻 _warp_stmatrix_global 的索引映射）
  实现 silu 就地变换 / rwv 乘法 + select

权重 tile 每 k-chunk 只从 global 读一次，供整个 256 行 tile 使用 -> 权重遍数减半。
single 类 kernel (G_S/U_S/D_S) 沿用 v22 已验证的 T.gemm + covered 谓词 (@th256)。

合并谓词（设备侧，穷举验证互斥完备）：
pair p=(2i,2i+1) 可合并 <=> 2i+1 < nbm 且 gidx[2i]==gidx[2i+1]
"""
import torch
import tilelang
import tilelang.language as T
# ===== classless MACA MFMA helpers (sandbox-safe) =====
def make_mfma_swizzle_layout(shared_buf):
    shape = shared_buf.shape

    def transform(row, col):
        # Official 128-byte XOR swizzle for fp16 K=64 shared tiles.
        phase = row % 8
        return row, ((col // 8) ^ phase) * 8 + col % 8

    return T.Layout(shape, transform)




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
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32

    num_pairs = (num_blocks_m + 1) // 2

    # ---- 手工 MMA 配置（官方 example_gemm_intrinsics.py 风格）----
    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    chunk = 64
    m_threads = 64 * (block_row_warps * block_col_warps)  # 512

    m_block_M = block_row_warps * warp_row_tiles  # 256
    m_block_N = block_col_warps * warp_col_tiles  # 128

    micro_size = 16
    warp_rows = warp_row_tiles // micro_size  # 4
    warp_cols = warp_col_tiles // micro_size  # 4
    local_size_out = (micro_size * micro_size) // 64  # 4
    n_ki = chunk // micro_size  # 2

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
        # ---- G_M: gate GEMM, 手工 MMA 合并 256x128 ----
        with T.Kernel(num_pairs, T.ceildiv(intermediate, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_mfma_swizzle_layout(A_shared),
                    B_shared: make_mfma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))

            if active == 1:
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(hidden, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = stacked_expert_tokens[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = gate_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    T.sync_threads()
                    for ki in T.serial(0, n_ki):
                        tx = T.get_thread_binding()
                        lane = tx % 64
                        warp_m = (tx // 64) % block_row_warps
                        warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                        for wi in T.serial(warp_rows):
                            for lid in T.vectorized(4):
                                A_local[wi * 4 + lid] = A_shared[
                                    warp_m * warp_row_tiles + wi * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wj in T.serial(warp_cols):
                            for lid in T.vectorized(4):
                                B_local[wj * 4 + lid] = B_shared[
                                    warp_n * warp_col_tiles + wj * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wi, wj in T.grid(warp_rows, warp_cols):
                            T.tvm_mfma(
                                "16x16x16f16", "row", "row",
                                "float16x4", "float16x4", "float32x4",
                                B_local.data, wj,
                                A_local.data, wi,
                                C_local.data, wi * warp_cols + wj,
                                dtype="float32x4",
                            )
                    T.sync_threads()

                tx = T.get_thread_binding()
                lane = tx % 64
                warp_m = (tx // 64) % block_row_warps
                warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                for wi, wj in T.grid(warp_rows, warp_cols):
                    for lid in T.vectorized(local_size_out):
                        row = lane % 16
                        col = lid + (lane // 16) * 4
                        ws[
                            block_start + (warp_m * warp_rows + wi) * micro_size + row,
                            by * m_block_N + (warp_n * warp_cols + wj) * micro_size + col,
                        ] = C_local[wi * warp_cols * local_size_out + wj * local_size_out + lid]

        # ---- U_M: up GEMM + 就地 silu, 手工 MMA 合并 ----
        with T.Kernel(num_pairs, T.ceildiv(intermediate, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_mfma_swizzle_layout(A_shared),
                    B_shared: make_mfma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))

            if active == 1:
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(hidden, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = stacked_expert_tokens[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = up_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    T.sync_threads()
                    for ki in T.serial(0, n_ki):
                        tx = T.get_thread_binding()
                        lane = tx % 64
                        warp_m = (tx // 64) % block_row_warps
                        warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                        for wi in T.serial(warp_rows):
                            for lid in T.vectorized(4):
                                A_local[wi * 4 + lid] = A_shared[
                                    warp_m * warp_row_tiles + wi * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wj in T.serial(warp_cols):
                            for lid in T.vectorized(4):
                                B_local[wj * 4 + lid] = B_shared[
                                    warp_n * warp_col_tiles + wj * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wi, wj in T.grid(warp_rows, warp_cols):
                            T.tvm_mfma(
                                "16x16x16f16", "row", "row",
                                "float16x4", "float16x4", "float32x4",
                                B_local.data, wj,
                                A_local.data, wi,
                                C_local.data, wi * warp_cols + wj,
                                dtype="float32x4",
                            )
                    T.sync_threads()

                # 自定义 fragment store：ws = silu(ws) * C_local（复刻 stmatrix global 索引映射）
                tx = T.get_thread_binding()
                lane = tx % 64
                warp_m = (tx // 64) % block_row_warps
                warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                for i, j in T.grid(warp_rows, warp_cols):
                    for local_id in T.vectorized(local_size_out):
                        row = lane % 16
                        col = local_id + (lane // 16) * 4
                        R = block_start + (warp_m * warp_rows + i) * micro_size + row
                        Cc = by * m_block_N + (warp_n * warp_cols + j) * micro_size + col
                        ws[R, Cc] = (
                            ws[R, Cc]
                            * (1.0 / (1.0 + T.exp2(-ws[R, Cc] * scale)))
                            * C_local[i * (warp_cols * local_size_out) + j * local_size_out + local_id]
                        )

        # ---- D_M: down GEMM, 手工 MMA 合并, rwv select ----
        with T.Kernel(num_pairs, T.ceildiv(hidden, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)
            rwv = T.alloc_shared((m_block_M,), dtype=T.float32)

            T.annotate_layout(
                {
                    A_shared: make_mfma_swizzle_layout(A_shared),
                    B_shared: make_mfma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))
            rw_last = total_valid_tokens - 1

            if active == 1:
                for i in T.Parallel(m_block_M):
                    rwv[i] = T.if_then_else(
                        i < actual_rows,
                        routed_expert_weights[T.min(raw_start + token_offset + i, rw_last)],
                        0.0,
                    )

                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(intermediate, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = ws[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = down_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    T.sync_threads()
                    for ki in T.serial(0, n_ki):
                        tx = T.get_thread_binding()
                        lane = tx % 64
                        warp_m = (tx // 64) % block_row_warps
                        warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                        for wi in T.serial(warp_rows):
                            for lid in T.vectorized(4):
                                A_local[wi * 4 + lid] = A_shared[
                                    warp_m * warp_row_tiles + wi * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wj in T.serial(warp_cols):
                            for lid in T.vectorized(4):
                                B_local[wj * 4 + lid] = B_shared[
                                    warp_n * warp_col_tiles + wj * micro_size + lane % 16,
                                    ki * micro_size + (lane // 16) * 4 + lid,
                                ]
                        for wi, wj in T.grid(warp_rows, warp_cols):
                            T.tvm_mfma(
                                "16x16x16f16", "row", "row",
                                "float16x4", "float16x4", "float32x4",
                                B_local.data, wj,
                                A_local.data, wi,
                                C_local.data, wi * warp_cols + wj,
                                dtype="float32x4",
                            )
                    T.sync_threads()

                tx = T.get_thread_binding()
                lane = tx % 64
                warp_m = (tx // 64) % block_row_warps
                warp_n = (tx // (64 * block_row_warps)) % block_col_warps
                for i, j in T.grid(warp_rows, warp_cols):
                    for local_id in T.vectorized(local_size_out):
                        row = lane % 16
                        col = local_id + (lane // 16) * 4
                        R = block_start + (warp_m * warp_rows + i) * micro_size + row
                        Cc = by * m_block_N + (warp_n * warp_cols + j) * micro_size + col
                        out[R, Cc] = T.if_then_else(
                            R - block_start < actual_rows,
                            C_local[i * (warp_cols * local_size_out) + j * local_size_out + local_id]
                            * rwv[R - block_start],
                            0.0,
                        )

        # ---- G_S: gate GEMM single (T.gemm, v22 已验证) ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
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

        # ---- U_S: up GEMM + 就地 silu single ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
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

        # ---- D_S: down GEMM single ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, block_n2), threads=threads_single) as (bx, by):
            hs = T.alloc_shared((block_token, block_k2), dtype=dtype)
            ds = T.alloc_shared((block_n2, block_k2), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n2), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
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


