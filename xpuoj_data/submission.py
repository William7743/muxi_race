# XPU-OJ v404 candidate: v399 + hidden7168-only Stage2 Down next-K prefetch
#
# 动机（对齐主线 PROGRESS/共享结论）：
#   - v380（全局 safe-off + vec256-off）已 2A，但全局关闭 safe pass 后偶发稀疏 WA；
#   - Stage2 唯一非 padded buffer 是 routed_expert_weights（raw 索引，在
#     `if i < actual_rows` 内访问），怀疑偶发 WA 来自 Stage2 尾块该 load 被
#     向量化/推测执行；
#   - safe-off 的 9~11% 收益预计主要来自占 ~78% 时间的 Stage1
#     （其输入/权重为完整矩阵、token padded 到 128，关闭语义安全）。
#
# 本版改动（其余 tile/threads/swizzle/GEMM/SwiGLU/索引语义与 v380 完全一致）：
#   Stage1 JIT: disable_safe_memory_legalize=True + disable_vectorize_256=True
#   Stage2 JIT: 保留默认 safe-memory legalize（只关 warp-specialized）
#   双 kernel 拆成两个独立 JIT callable，仍为同流两次 GPU launch。
import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
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
            up_prefetch = T.alloc_fragment((be1, bh1), dtype=dtype)
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
                T.gemm(up_shared, down_shared, out_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)

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
    }
)
def _moe_stage2_prefetch(
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
    """Stage2 with synchronous next-K Down-weight register prefetch.

    This is a distinct hidden=7168-only JIT so the sensitive case1 lowering
    remains byte-for-byte on the original v399 Stage2 function.  The prefetch
    is global -> fragment -> shared and does not use async/bsm instructions.
    """
    dtype = T.float16
    accum_dtype = T.float32
    k_steps = intermediate // be2
    last_k = k_steps - 1

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
            down_prefetch = T.alloc_fragment((bh2, be2), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(bt1, group_size - token_offset))

            T.clear(out_local)

            # Prime k=0.  Each group_idx entry represents a block containing
            # at least one valid token, so every launched CTA has k_steps work.
            T.copy(
                down_w[
                    expert_id,
                    by * bh2 : (by + 1) * bh2,
                    0:be2,
                ],
                down_prefetch,
                coalesced_width=8,
            )

            # After the current prefetched tile is committed to shared, issue
            # the next global load before MMA.  Its scoreboard wait is delayed
            # until the following iteration, overlapping it with current MMA.
            for k in range(k_steps - 1):
                T.copy(
                    up_logits[
                        block_start : block_start + bt1,
                        k * be2 : (k + 1) * be2,
                    ],
                    up_shared,
                )
                T.copy(down_prefetch, down_shared, coalesced_width=8)
                T.copy(
                    down_w[
                        expert_id,
                        by * bh2 : (by + 1) * bh2,
                        (k + 1) * be2 : (k + 2) * be2,
                    ],
                    down_prefetch,
                    coalesced_width=8,
                )
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
                    last_k * be2 : (last_k + 1) * be2,
                ],
                up_shared,
            )
            T.copy(down_prefetch, down_shared, coalesced_width=8)
            T.gemm(
                up_shared,
                down_shared,
                out_local,
                transpose_B=True,
                policy=T.GemmWarpPolicy.Square,
            )

            for i, j in T.Parallel(bt1, bh2):
                if i < actual_rows:
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
                else:
                    out[block_start + i, by * bh2 + j] = 0

    return stage2


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
        builder = _moe_stage1_prefetch if hidden >= 7000 else _moe_stage1
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
        builder = _moe_stage2_prefetch if hidden >= 7000 else _moe_stage2
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
