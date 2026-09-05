# CASES=1,2,3
# v749 experimental: v745 + E16/H2048/I8192 runtime M128/M64 Stage1.
# One Stage1 T.Kernel and the original Input/Gate/Up prefetch loop remain.
# Full rows>64 retains the parent active_k_steps loop; 0<rows<=64 uses
# the first64-row A shared view and separate64x128 Gate/Up accumulators.
# A128x64 + one B128x64 shared pair remains32KiB; current-K Up prefetch
# is overwritten on every iteration. Empty/invalid rows leave workspace untouched.
# No GIU reorder, terminal-K change, new flags, extra launch/workspace or async.
# E32/E64/Stage2/host behavior remains v745; no cross-call result reuse.
# STATIC CANDIDATE ONLY: not compiled, not GPU tested, not OJ tested.
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
    }
)
def _moe_stage2_fast_bfrag_prefetch_route_bounds(
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
    """E32 nonempty-route memory-safety probe; clamp only route-load addresses."""
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
                        out_local[i, j] * routed_expert_weights[
                            T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))
                        ]
                    )
            else:
                for i, j in T.Parallel(bt1, bh2):
                    if i < actual_rows:
                        out[block_start + i, by * bh2 + j] = (
                            out_local[i, j] * routed_expert_weights[
                                T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))
                            ]
                        )
                    else:
                        out[block_start + i, by * bh2 + j] = 0

    return stage2


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def _moe_stage2_e32_zero_output(
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
    """Compile-time empty-route path: no input reads, including no route loads."""
    dtype = T.float16
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
        with T.Kernel(
            T.ceildiv(total_padded_tokens, bt1), T.ceildiv(hidden, bh2), threads=th2
        ) as (bx, by):
            for i, j in T.Parallel(bt1, bh2):
                if bx * bt1 + i < total_padded_tokens and by * bh2 + j < hidden:
                    out[bx * bt1 + i, by * bh2 + j] = 0

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
    }
)
def _moe_stage2_runtime_m64_route_bounds(
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
    """One E32 Stage2 launch; uniform M128/M64/zero paths with clamped routes."""
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
    tail_m = 64
    tail_mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=64,
        chunk=be2,
        k_pack=1,
    )
    tail_a_local_size = tail_mma_emitter.warp_rows * tail_mma_emitter.local_size_a
    tail_b_local_size = tail_mma_emitter.warp_cols * tail_mma_emitter.local_size_b

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
            # Independent private layouts; both branches share only A/B shared.
            tail_up_matrix = T.alloc_fragment((tail_a_local_size,), dtype=dtype)
            tail_down_matrix0 = T.alloc_fragment((tail_b_local_size,), dtype=dtype)
            tail_down_matrix1 = T.alloc_fragment((tail_b_local_size,), dtype=dtype)
            tail_out_local = T.alloc_fragment((tail_m, bh2), dtype=accum_dtype)

            T.annotate_layout(
                {
                    up_shared: make_mma_swizzle_layout(up_shared, vecSize=4),
                    down_shared: make_mma_swizzle_layout(down_shared, vecSize=4),
                    out_local: mma_emitter.make_mma_store_layout(out_local),
                    tail_out_local: tail_mma_emitter.make_mma_store_layout(tail_out_local),
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

            # CTA-uniform row-count dispatch: never changes launch geometry.
            # Keep the full M128 path intact; tail copies/reads only A's first64.
            if actual_rows > tail_m:
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
                            out_local[i, j] * routed_expert_weights[
                                T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))
                            ]
                        )
                else:
                    for i, j in T.Parallel(bt1, bh2):
                        if i < actual_rows:
                            out[block_start + i, by * bh2 + j] = (
                                out_local[i, j] * routed_expert_weights[
                                    T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))
                                ]
                            )
                        else:
                            out[block_start + i, by * bh2 + j] = 0
            elif actual_rows > 0:
                T.clear(tail_out_local)
                T.copy(
                    up_logits[
                        block_start : block_start + tail_m,
                        0:be2,
                    ],
                    up_shared[0:tail_m, 0:be2],
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * bh2 : (by + 1) * bh2,
                        0:be2,
                    ],
                    down_shared,
                )

                for k in range(T.ceildiv(intermediate, be2) - 1):
                    tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 0)
                    tail_mma_emitter.ldmatrix_b(tail_down_matrix0, down_shared, 0)
                    tail_mma_emitter.ldmatrix_b(tail_down_matrix1, down_shared, 1)
                    tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix0, tail_out_local)
                    tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 1)
                    tail_mma_emitter.ldmatrix_b(tail_down_matrix0, down_shared, 2)
                    tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix1, tail_out_local)
                    tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 2)
                    tail_mma_emitter.ldmatrix_b(tail_down_matrix1, down_shared, 3)
                    tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix0, tail_out_local)
                    tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 3)
                    tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix1, tail_out_local)
                    T.sync_threads()
                    T.copy(
                        up_logits[
                            block_start : block_start + tail_m,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        up_shared[0:tail_m, 0:be2],
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * bh2 : (by + 1) * bh2,
                            (k + 1) * be2 : (k + 2) * be2,
                        ],
                        down_shared,
                    )

                tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 0)
                tail_mma_emitter.ldmatrix_b(tail_down_matrix0, down_shared, 0)
                tail_mma_emitter.ldmatrix_b(tail_down_matrix1, down_shared, 1)
                tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix0, tail_out_local)
                tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 1)
                tail_mma_emitter.ldmatrix_b(tail_down_matrix0, down_shared, 2)
                tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix1, tail_out_local)
                tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 2)
                tail_mma_emitter.ldmatrix_b(tail_down_matrix1, down_shared, 3)
                tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix0, tail_out_local)
                tail_mma_emitter.ldmatrix_a(tail_up_matrix, up_shared[0:tail_m, 0:be2], 3)
                tail_mma_emitter.mma(tail_up_matrix, tail_down_matrix1, tail_out_local)

                for i, j in T.Parallel(tail_m, bh2):
                    if i < actual_rows:
                        out[block_start + i, by * bh2 + j] = (
                            tail_out_local[i, j]
                            * routed_expert_weights[
                                T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))
                            ]
                        )
                    else:
                        out[block_start + i, by * bh2 + j] = 0

                for i, j in T.Parallel(tail_m, bh2):
                    out[block_start + tail_m + i, by * bh2 + j] = 0
            else:
                for i, j in T.Parallel(bt1, bh2):
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
def _moe_stage1_runtime_m64_giu_merge(
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
    tail_m = 64
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
        # One A128x64 + one reusable B128x64 shared pair = 32KiB.
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            tail_gate_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)
            tail_up_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)

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

            # Uniform CTA choice; each branch owns its G/U accumulators.
            # Shared A retains its128-row allocation/layout; tail uses a64-row view.
            if actual_rows > tail_m:
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
            elif actual_rows > 0:
                T.clear(tail_gate_local)
                T.clear(tail_up_local)

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
                                block_start : block_start + tail_m,
                                k * bh1 : (k + 1) * bh1,
                            ],
                            input_shared[0:tail_m, 0:bh1],
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
                        T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                        T.sync_threads()
                        T.copy(
                            up_prefetch,
                            weight_shared,
                            coalesced_width=4,
                        )
                        T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
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
                            block_start : block_start + tail_m,
                            terminal_k * bh1 : (terminal_k + 1) * bh1,
                        ],
                        input_shared[0:tail_m, 0:bh1],
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
                    T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()
                    T.copy(
                        up_prefetch,
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)

                for i, j in T.Parallel(tail_m, be1):
                    # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                    # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                    if i < actual_rows:
                        up_logits[block_start + i, by * be1 + j] = (
                            tail_up_local[i, j]
                            * (
                                tail_gate_local[i, j]
                                * (1.0 / (1.0 + T.exp2(-tail_gate_local[i, j] * scale)))
                            )
                        )
            # Empty blocks intentionally leave workspace untouched, as v743.

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
def _moe_stage1_e16_runtime_m64_prefetch(
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
    tail_m = 64
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
        # One A128x64 + one reusable B128x64 shared pair = 32KiB.
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            tail_gate_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)
            tail_up_local = T.alloc_fragment((tail_m, be1), dtype=accum_dtype)

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

            # CTA-uniform selection; no shared access or workspace store for empty blocks.
            if actual_rows > tail_m:
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
            elif actual_rows > 0:
                T.clear(tail_gate_local)
                T.clear(tail_up_local)

                # A normal serial loop permits the Gate and Up tiles to reuse one
                # shared allocation.  Explicit barriers protect the overwrite
                # while the other waves may still be consuming the prior tile.
                for k in range(active_k_steps):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + tail_m,
                            k * bh1 : (k + 1) * bh1,
                        ],
                        input_shared[0:tail_m, 0:bh1],
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
                    T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()
                    T.copy(
                        up_prefetch,
                        weight_shared,
                        coalesced_width=4,
                    )
                    T.gemm(input_shared[0:tail_m, 0:bh1], weight_shared, tail_up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                    T.sync_threads()

                for i, j in T.Parallel(tail_m, be1):
                    # 仅写有效行：padding 行的 stacked 输入是任意值，写出来也无意义，
                    # Stage2 会用 else 分支把 padding 行输出清 0；跳过实测快 14%
                    if i < actual_rows:
                        up_logits[block_start + i, by * be1 + j] = (
                            tail_up_local[i, j]
                            * (
                                tail_gate_local[i, j]
                                * (1.0 / (1.0 + T.exp2(-tail_gate_local[i, j] * scale)))
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
            (
                _moe_stage1_runtime_m64_giu_merge
                if hidden == 7168 and intermediate == 2048
                and total_padded_tokens > 0 and num_blocks_m > 0
                else _moe_stage1_prefetch_giu_merge
            )
            if num_experts == 32
            else (
                _moe_stage1_prefetch_giu_merge
                if num_experts == 64
                else (
                    _moe_stage1_e16_runtime_m64_prefetch
                    if num_experts == 16 and hidden == 2048 and intermediate == 8192
                    and total_padded_tokens > 0 and num_blocks_m > 0
                    else _moe_stage1_prefetch
                )
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
        if num_experts == 32:
            # Empty arrays have no valid clamped address; select a no-load kernel.
            builder = (
                _moe_stage2_e32_zero_output
                if total_valid_tokens == 0
                else (
                    _moe_stage2_runtime_m64_route_bounds
                    if hidden == 7168 and intermediate == 2048
                    and total_valid_tokens > 0 and total_padded_tokens > 0 and num_blocks_m > 0
                    else _moe_stage2_fast_bfrag_prefetch_route_bounds
                )
            )
        else:
            builder = (
                _moe_stage2_fast_bfrag_prefetch
                if num_experts in (16, 64)
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

    if num_experts == 32 and total_padded_tokens == 0:
        return

    up_logits = _get_workspace(stacked_expert_tokens, intermediate)
    # routed_expert_weights 的 dtype 题面写 fp16 但参考实现存在 fp32 声明，
    # 这里按实际传入 dtype 编译，两种都兼容
    if routed_expert_weights.dtype == torch.float32:
        weights_dtype = T.float32
    else:
        weights_dtype = T.float16

    if num_experts == 32 and total_valid_tokens == 0:
        # No Stage1 work is needed: Stage2 overwrites the complete padded output.
        stage2 = _get_stage2(
            hidden,
            intermediate,
            num_experts,
            total_padded_tokens,
            total_valid_tokens,
            num_blocks_m,
            weights_dtype,
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
        return

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
