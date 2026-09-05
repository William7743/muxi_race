# CASES=1,2,3
# XPU-OJ v724: v720 with E32-only Stage1 retaining four A microfragments across Gate/Up.
# Single B shared tile, GIU global copies, Up register prefetch and terminal K are retained.
# Official v563 single-B emitter geometry/layout is reused, with four persistent A fragments.
# Up MMA reuses the current K's A operands instead of loading input_shared a second time.
# Shared storage remains 32 KiB; added A register lifetime may cause pressure or spilling.
# E16/E64 and all Stage2 paths remain v720; current inputs are recomputed in two launches.
# Local random checks passed; 0.19% short-timing difference is inconclusive. No OJ test yet.
# No async/BSM, pipeline DSL, external code, or result caching.
import torch
import tilelang
import tilelang.language as T
from tilelang.maca.intrinsics import TensorCoreIntrinEmitter, make_mma_swizzle_layout


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
    }
)
def _moe_stage1(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bt1,
    bh1,
    be1,
    th1,
):
    gu_k_pack = 2 if hidden >= 7000 else 1
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)

    @T.prim_func
    def stage1(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace ----
        # smem: A(bt1*bh1) + gate(be1*bh1) + up(be1*bh1) = (128+256)*64*2B = 48KB
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            # swizzle(4)：OJ 三用例实测比默认 swizzle(10) 稳定快 ~0.7%
            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)

            T.clear(gate_local)
            T.clear(up_local)

            # A normal serial loop permits the Gate and Up tiles to reuse one
            # shared allocation.  Explicit barriers protect the overwrite
            # while the other waves may still be consuming the prior tile.
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
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        k * bh1 : (k + 1) * bh1,
                    ],
                    weight_shared,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(bt1, be1):
                # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return stage1


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
    }
)
def _moe_stage1_prefetch(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bt1,
    bh1,
    be1,
    th1,
):
    gu_k_pack = 2
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)

    @T.prim_func
    def stage1(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace ----
        # smem: A(bt1*bh1) + gate(be1*bh1) + up(be1*bh1) = (128+256)*64*2B = 48KB
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.annotate_layout(
                {
                    input_shared: make_mma_swizzle_layout(input_shared, vecSize=4),
                    weight_shared: make_mma_swizzle_layout(weight_shared, vecSize=4),
                }
            )

            # swizzle(4)：OJ 三用例实测比默认 swizzle(10) 稳定快 ~0.7%
            T.use_swizzle(3 if num_experts == 32 else 2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)

            T.clear(gate_local)
            T.clear(up_local)

            # A normal serial loop permits the Gate and Up tiles to reuse one
            # shared allocation.  Explicit barriers protect the overwrite
            # while the other waves may still be consuming the prior tile.
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
                T.copy(
                    up_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        k * bh1 : (k + 1) * bh1,
                    ],
                    up_prefetch,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_prefetch,
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(bt1, be1):
                # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return stage1




@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.enable_fast_math": True,
    }
)
def _moe_stage2(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    bt1,
    bh2,
    be2,
    th2,
    weights_dtype,
):
    dtype = T.float16
    accum_dtype = T.float32

    intermediate_shape = (total_padded_tokens, intermediate)
    input_shape = (total_padded_tokens, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def stage2(
        up_logits: T.Tensor(intermediate_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Stage2: down GEMM × routed_weight -> out（padding 行写 0）----
        # smem: A(bt1*be2) + down(bh2*be2) = (128+128)*64*2B = 32KB
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, bh2), threads=th2) as (bx, by):
            up_shared = T.alloc_shared((bt1, be2), dtype=dtype)
            down_shared = T.alloc_shared((bh2, be2), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(bt1, group_size - token_offset))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)

            T.clear(out_local)

            if active_k_steps > 0:
                T.copy(
                    up_logits[block_start : block_start + bt1, 0:be2],
                    up_shared,
                )
                T.copy(
                    down_w[expert_id, by * bh2 : (by + 1) * bh2, 0:be2],
                    down_shared,
                )
                for k in range(active_k_steps - 1):
                    T.gemm(
                        up_shared,
                        down_shared,
                        out_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.Square,
                    )
                    T.sync_threads()
                    T.copy(
                        up_logits[
                            block_start : block_start + bt1,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * bh2 : (by + 1) * bh2,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        down_shared,
                    )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.Square,
                )

            for i, j in T.Parallel(bt1, bh2):
                if i < actual_rows:
                    # routed_expert_weights 按真实 token 顺序索引（raw 坐标）
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
                else:
                    out[block_start + i, by * bh2 + j] = 0

    return stage2


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
    }
)
def _moe_stage2_fast(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    bt1,
    bh2,
    be2,
    th2,
    weights_dtype,
):
    """Byte-equivalent Stage2 body with fast passes for hidden=7168 only."""
    dtype = T.float16
    accum_dtype = T.float32

    intermediate_shape = (total_padded_tokens, intermediate)
    input_shape = (total_padded_tokens, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def stage2(
        up_logits: T.Tensor(intermediate_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, bh2), threads=th2) as (bx, by):
            up_shared = T.alloc_shared((bt1, be2), dtype=dtype)
            down_shared = T.alloc_shared((bh2, be2), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.annotate_layout(
                {
                    up_shared: make_mma_swizzle_layout(up_shared, vecSize=4),
                    down_shared: make_mma_swizzle_layout(down_shared, vecSize=4),
                }
            )

            T.use_swizzle(2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(bt1, group_size - token_offset))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)

            T.clear(out_local)

            if active_k_steps > 0:
                T.copy(
                    up_logits[block_start : block_start + bt1, 0:be2],
                    up_shared,
                )
                T.copy(
                    down_w[expert_id, by * bh2 : (by + 1) * bh2, 0:be2],
                    down_shared,
                )
                for k in range(active_k_steps - 1):
                    T.gemm(
                        up_shared,
                        down_shared,
                        out_local,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.Square,
                    )
                    T.sync_threads()
                    T.copy(
                        up_logits[
                            block_start : block_start + bt1,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * bh2 : (by + 1) * bh2,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        down_shared,
                    )
                T.gemm(
                    up_shared,
                    down_shared,
                    out_local,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.Square,
                )

            if actual_rows == bt1:
                for i, j in T.Parallel(bt1, bh2):
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
            else:
                for i, j in T.Parallel(bt1, bh2):
                    if i < actual_rows:
                        out[block_start + i, by * bh2 + j] = (
                            out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                        )
                    else:
                        out[block_start + i, by * bh2 + j] = 0

    return stage2


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
        "tl.enable_aggressive_shared_memory_merge": True,
    }
)
def _moe_stage1_prefetch_giu_merge(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bt1,
    bh1,
    be1,
    th1,
):
    gu_k_pack = 2
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)

    @T.prim_func
    def stage1(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace ----
        # smem: A(bt1*bh1) + gate(be1*bh1) + up(be1*bh1) = (128+256)*64*2B = 48KB
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.annotate_layout(
                {
                    input_shared: make_mma_swizzle_layout(input_shared, vecSize=4),
                    weight_shared: make_mma_swizzle_layout(weight_shared, vecSize=4),
                }
            )

            # swizzle(4)：OJ 三用例实测比默认 swizzle(10) 稳定快 ~0.7%
            T.use_swizzle(3 if num_experts == 32 else 2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
            k_steps = T.ceildiv(hidden, bh1)

            T.clear(gate_local)
            T.clear(up_local)

            # A normal serial loop permits the Gate and Up tiles to reuse one
            # shared allocation.  Explicit barriers protect the overwrite
            # while the other waves may still be consuming the prior tile.
            if actual_rows > 0:
                for k in range(k_steps - 1):
                    T.copy(
                        gate_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + bt1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        input_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        up_prefetch,
                        coalesced_width=8,
                    )
                    T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()
                    T.copy(
                        up_prefetch,
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()

                terminal_k = k_steps - 1
                T.copy(
                    gate_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + bt1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    input_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    up_prefetch,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_prefetch,
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)

            for i, j in T.Parallel(bt1, be1):
                # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return stage1


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
    }
)
def _moe_stage2_fast_bfrag_prefetch(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    bt1,
    bh2,
    be2,
    th2,
    weights_dtype,
):
    """E32 Stage2 with direct MMA and two B fragments for LDS latency hiding."""
    dtype = T.float16
    accum_dtype = T.float32
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=be2,
        k_pack=1,
    )
    a_local_size = mma_emitter.warp_rows * mma_emitter.local_size_a
    b_local_size = mma_emitter.warp_cols * mma_emitter.local_size_b

    intermediate_shape = (total_padded_tokens, intermediate)
    input_shape = (total_padded_tokens, hidden)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def stage2(
        up_logits: T.Tensor(intermediate_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        out: T.Tensor(input_shape, dtype),
    ):
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, bh2), threads=th2) as (bx, by):
            up_shared = T.alloc_shared((bt1, be2), dtype=dtype)
            down_shared = T.alloc_shared((bh2, be2), dtype=dtype)
            up_matrix = T.alloc_fragment((a_local_size,), dtype=dtype)
            down_matrix0 = T.alloc_fragment((b_local_size,), dtype=dtype)
            down_matrix1 = T.alloc_fragment((b_local_size,), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.annotate_layout(
                {
                    up_shared: make_mma_swizzle_layout(up_shared, vecSize=4),
                    down_shared: make_mma_swizzle_layout(down_shared, vecSize=4),
                    out_local: mma_emitter.make_mma_store_layout(out_local),
                }
            )

            T.use_swizzle(2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(bt1, group_size - token_offset))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)

            if active_k_steps > 0:
                T.copy(
                    up_logits[block_start : block_start + bt1, 0:be2],
                    up_shared,
                )
                T.clear(out_local)
                T.copy(
                    down_w[expert_id, by * bh2 : (by + 1) * bh2, 0:be2],
                    down_shared,
                )
                for k in range(active_k_steps - 1):
                    mma_emitter.ldmatrix_a(up_matrix, up_shared, 0)
                    mma_emitter.ldmatrix_b(down_matrix0, down_shared, 0)
                    mma_emitter.ldmatrix_b(down_matrix1, down_shared, 1)
                    mma_emitter.mma(up_matrix, down_matrix0, out_local)
                    mma_emitter.ldmatrix_a(up_matrix, up_shared, 1)
                    mma_emitter.ldmatrix_b(down_matrix0, down_shared, 2)
                    mma_emitter.mma(up_matrix, down_matrix1, out_local)
                    mma_emitter.ldmatrix_a(up_matrix, up_shared, 2)
                    mma_emitter.ldmatrix_b(down_matrix1, down_shared, 3)
                    mma_emitter.mma(up_matrix, down_matrix0, out_local)
                    mma_emitter.ldmatrix_a(up_matrix, up_shared, 3)
                    mma_emitter.mma(up_matrix, down_matrix1, out_local)
                    T.sync_threads()
                    T.copy(
                        up_logits[
                            block_start : block_start + bt1,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        up_shared,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * bh2 : (by + 1) * bh2,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        down_shared,
                    )
                mma_emitter.ldmatrix_a(up_matrix, up_shared, 0)
                mma_emitter.ldmatrix_b(down_matrix0, down_shared, 0)
                mma_emitter.ldmatrix_b(down_matrix1, down_shared, 1)
                mma_emitter.mma(up_matrix, down_matrix0, out_local)
                mma_emitter.ldmatrix_a(up_matrix, up_shared, 1)
                mma_emitter.ldmatrix_b(down_matrix0, down_shared, 2)
                mma_emitter.mma(up_matrix, down_matrix1, out_local)
                mma_emitter.ldmatrix_a(up_matrix, up_shared, 2)
                mma_emitter.ldmatrix_b(down_matrix1, down_shared, 3)
                mma_emitter.mma(up_matrix, down_matrix0, out_local)
                mma_emitter.ldmatrix_a(up_matrix, up_shared, 3)
                mma_emitter.mma(up_matrix, down_matrix1, out_local)

            if actual_rows == bt1:
                for i, j in T.Parallel(bt1, bh2):
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
            else:
                for i, j in T.Parallel(bt1, bh2):
                    if i < actual_rows:
                        out[block_start + i, by * bh2 + j] = (
                            out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                        )
                    else:
                        out[block_start + i, by * bh2 + j] = 0

    return stage2


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
        "tl.enable_aggressive_shared_memory_merge": True,
    }
)
def _moe_stage1_prefetch_giu_merge_v527(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bt1,
    bh1,
    be1,
    th1,
):
    gu_k_pack = 2
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)

    @T.prim_func
    def stage1(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace ----
        # smem: A(bt1*bh1) + gate(be1*bh1) + up(be1*bh1) = (128+256)*64*2B = 48KB
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.annotate_layout(
                {
                    input_shared: make_mma_swizzle_layout(input_shared, vecSize=4),
                    weight_shared: make_mma_swizzle_layout(weight_shared, vecSize=4),
                }
            )

            # swizzle(4)：OJ 三用例实测比默认 swizzle(10) 稳定快 ~0.7%
            T.use_swizzle(3 if num_experts == 32 else 2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)

            T.clear(gate_local)
            T.clear(up_local)

            # A normal serial loop permits the Gate and Up tiles to reuse one
            # shared allocation.  Explicit barriers protect the overwrite
            # while the other waves may still be consuming the prior tile.
            for k in range(active_k_steps):
                T.copy(
                    gate_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        k * bh1 : (k + 1) * bh1,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + bt1,
                        k * bh1 : (k + 1) * bh1,
                    ],
                    input_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        k * bh1 : (k + 1) * bh1,
                    ],
                    up_prefetch,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_prefetch,
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(bt1, be1):
                # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return stage1


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
        "tl.enable_fast_math": True,
        "tl.enable_lower_ldgstg_predicated": True,
        "tl.enable_aggressive_shared_memory_merge": True,
    }
)
def _moe_stage1_prefetch_giu_merge_a_reuse(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bt1,
    bh1,
    be1,
    th1,
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=64,
        warp_col_tiles=64,
        chunk=bh1,
        k_pack=1,
    )
    a_local_size = mma_emitter.warp_rows * mma_emitter.k_pack * mma_emitter.local_size_a
    b_local_size = mma_emitter.warp_cols * mma_emitter.k_pack * mma_emitter.local_size_b

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)

    @T.prim_func
    def stage1(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace ----
        # Shared: one A tile + one reused Gate/Up tile = 2 * 128 * 64 * 2 B = 32 KiB.
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            input_matrix0 = T.alloc_fragment((a_local_size,), dtype=dtype)
            input_matrix1 = T.alloc_fragment((a_local_size,), dtype=dtype)
            input_matrix2 = T.alloc_fragment((a_local_size,), dtype=dtype)
            input_matrix3 = T.alloc_fragment((a_local_size,), dtype=dtype)
            weight_matrix = T.alloc_fragment((b_local_size,), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.annotate_layout(
                {
                    input_shared: make_mma_swizzle_layout(input_shared, vecSize=4),
                    weight_shared: make_mma_swizzle_layout(weight_shared, vecSize=4),
                    gate_local: mma_emitter.make_mma_store_layout(gate_local),
                    up_local: mma_emitter.make_mma_store_layout(up_local),
                }
            )

            # swizzle(4)：OJ 三用例实测比默认 swizzle(10) 稳定快 ~0.7%
            T.use_swizzle(3 if num_experts == 32 else 2, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
            k_steps = T.ceildiv(hidden, bh1)

            T.clear(gate_local)
            T.clear(up_local)

            # A normal serial loop permits the Gate and Up tiles to reuse one
            # shared allocation.  Explicit barriers protect the overwrite
            # while the other waves may still be consuming the prior tile.
            if actual_rows > 0:
                for k in range(k_steps - 1):
                    T.copy(
                        gate_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + bt1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        input_shared,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * be1 : (by + 1) * be1,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        up_prefetch,
                        coalesced_width=8,
                    )
                    mma_emitter.ldmatrix_a(input_matrix0, input_shared, 0)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 0)
                    mma_emitter.mma(input_matrix0, weight_matrix, gate_local)
                    mma_emitter.ldmatrix_a(input_matrix1, input_shared, 1)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 1)
                    mma_emitter.mma(input_matrix1, weight_matrix, gate_local)
                    mma_emitter.ldmatrix_a(input_matrix2, input_shared, 2)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 2)
                    mma_emitter.mma(input_matrix2, weight_matrix, gate_local)
                    mma_emitter.ldmatrix_a(input_matrix3, input_shared, 3)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)
                    mma_emitter.mma(input_matrix3, weight_matrix, gate_local)
                    T.sync_threads()
                    T.copy(
                        up_prefetch,
                        weight_shared,
                        coalesced_width=4,
                    )
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 0)
                    mma_emitter.mma(input_matrix0, weight_matrix, up_local)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 1)
                    mma_emitter.mma(input_matrix1, weight_matrix, up_local)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 2)
                    mma_emitter.mma(input_matrix2, weight_matrix, up_local)
                    mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)
                    mma_emitter.mma(input_matrix3, weight_matrix, up_local)
                    T.sync_threads()

                terminal_k = k_steps - 1
                T.copy(
                    gate_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + bt1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    input_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * be1 : (by + 1) * be1,
                        terminal_k * bh1 : (terminal_k + 1) * bh1,
                    ],
                    up_prefetch,
                    coalesced_width=8,
                )
                mma_emitter.ldmatrix_a(input_matrix0, input_shared, 0)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 0)
                mma_emitter.mma(input_matrix0, weight_matrix, gate_local)
                mma_emitter.ldmatrix_a(input_matrix1, input_shared, 1)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 1)
                mma_emitter.mma(input_matrix1, weight_matrix, gate_local)
                mma_emitter.ldmatrix_a(input_matrix2, input_shared, 2)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 2)
                mma_emitter.mma(input_matrix2, weight_matrix, gate_local)
                mma_emitter.ldmatrix_a(input_matrix3, input_shared, 3)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)
                mma_emitter.mma(input_matrix3, weight_matrix, gate_local)
                T.sync_threads()
                T.copy(
                    up_prefetch,
                    weight_shared,
                    coalesced_width=4,
                )
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 0)
                mma_emitter.mma(input_matrix0, weight_matrix, up_local)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 1)
                mma_emitter.mma(input_matrix1, weight_matrix, up_local)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 2)
                mma_emitter.mma(input_matrix2, weight_matrix, up_local)
                mma_emitter.ldmatrix_b(weight_matrix, weight_shared, 3)
                mma_emitter.mma(input_matrix3, weight_matrix, up_local)

            for i, j in T.Parallel(bt1, be1):
                # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return stage1


def _pick_tiles(intermediate):
    # group_idx_for_bx 按 128 token/block 预计算，block_token 必须保持 128。
    # Stage1 首选 be=128/bh=64/threads=256（OJ 三用例实测最优）。
    return 128, 64, 128, 256  # v388 与 v380 保持一致


def _get_stage1(hidden, intermediate, num_experts, total_padded_tokens, num_blocks_m):
    bt1, bh1, be1, th1 = _pick_tiles(intermediate)
    key = (
        "s1",
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(num_blocks_m),
        bt1,
        bh1,
        be1,
        th1,
    )
    stage1 = _KERNEL_CACHE.get(key)
    if stage1 is None:
        builder = (
            _moe_stage1_prefetch_giu_merge_a_reuse
            if num_experts == 32
            else (
                _moe_stage1_prefetch_giu_merge
                if num_experts == 64
                else _moe_stage1_prefetch
            )
        )
        stage1 = builder(*key[1:])
        _KERNEL_CACHE[key] = stage1
    return stage1


def _get_stage2(
    hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, num_blocks_m, weights_dtype
):
    bt1 = _pick_tiles(intermediate)[0]
    bh2, be2, th2 = 128, 64, 256
    key = (
        "s2",
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
        int(bt1),
        bh2,
        be2,
        th2,
        str(weights_dtype),
    )
    stage2 = _KERNEL_CACHE.get(key)
    if stage2 is None:
        builder = (
            _moe_stage2_fast_bfrag_prefetch
            if num_experts in (16, 32, 64)
            else _moe_stage2_fast
        )
        stage2 = builder(*key[1:])
        _KERNEL_CACHE[key] = stage2
    return stage2


def _get_workspace(stacked_expert_tokens, intermediate):
    # up_logits workspace：按 padded 行数 × intermediate 缓存复用
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
    # routed_expert_weights 的 dtype 题面写 fp16 但参考实现存在 fp32 声明，
    # 这里按实际传入 dtype 编译，两种都兼容
    if routed_expert_weights.dtype == torch.float32:
        weights_dtype = T.float32
    else:
        weights_dtype = T.float16

    stage1 = _get_stage1(hidden, intermediate, num_experts, total_padded_tokens, num_blocks_m)
    stage2 = _get_stage2(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
        weights_dtype,
    )
    # 同流两次 launch：Stage2 依赖 Stage1 的 up_logits，默认流天然串行
    stage1(
        stacked_expert_tokens,
        gate_w,
        up_w,
        group_sizes,
        group_padded_offsets,
        group_idx_for_bx,
        up_logits,
    )
    stage2(
        up_logits,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        out,
    )
