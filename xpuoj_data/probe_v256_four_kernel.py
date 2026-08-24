"""
Probe v256: 4-Kernel structure (Gate/Up/Down/Routing split)
=========================================================
尝试将 MoE 计算拆分为 4 个独立的 kernel：
- Kernel 1: Gate GEMM (x @ gate_w^T)
- Kernel 2: Up GEMM (x @ up_w^T)
- Kernel 3: SiLU + Multiply (gate * sigmoid(gate) * up)
- Kernel 4: Down GEMM + Routing ((gate*sig*up) @ down_w^T * weights)
- 目标：减少每个 kernel 的复杂度，提高 occupancy
"""
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    gu_k_pack = 2 if hidden >= 7000 else 1
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

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
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        gate_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Kernel 1: Gate GEMM ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            input_shared = T.alloc_shared((128, 64), dtype=dtype)
            weight_shared = T.alloc_shared((128, 64), dtype=dtype)
            acc = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(128, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, 64), 0)

            T.clear(acc)
            for k in range(active_k_steps):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    input_shared,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * 128 : (by + 1) * 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(128, 128):
                if i < actual_rows:
                    gate_logits[block_start + i, by * 128 + j] = acc[i, j]

        # ---- Kernel 2: Up GEMM ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            input_shared = T.alloc_shared((128, 64), dtype=dtype)
            weight_shared = T.alloc_shared((128, 64), dtype=dtype)
            acc = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(128, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, 64), 0)

            T.clear(acc)
            for k in range(active_k_steps):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    input_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * 128 : (by + 1) * 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, acc, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(128, 128):
                if i < actual_rows:
                    gate_logits[block_start + i, by * 128 + j] = acc[i, j]  # 覆盖！

        # ---- Kernel 3: SiLU + Multiply (gate * sigmoid(gate) * up) ----
        # 注意：这里我们简化处理，假设 gate_logits 存储了 gate，而 up 结果存储在另一个 workspace
        # 由于内存限制，我们合并到 kernel 4 中

        # ---- Kernel 4: Down GEMM + SiLU fusion ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, 128), threads=256) as (bx, by):
            # 由于没有足够的 workspace，我们需要重新计算 or 使用临时存储
            # 这里简化为只计算 down GEMM，假设输入已经在某个地方
            pass

    return kernel


# 这个版本过于复杂，简化为与 v226 相同的结构但尝试不同的 fused 策略
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel_v2(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    gu_k_pack = 2 if hidden >= 7000 else 1
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

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
        routed_expert_weights: T.Tensor((total_valid_tokens,), T.float16),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        gate_up_logits: T.Tensor(intermediate_shape, dtype),  # 存储 gate 和 up
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Kernel 1: Gate GEMM only ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            input_shared = T.alloc_shared((128, 64), dtype=dtype)
            weight_shared = T.alloc_shared((128, 64), dtype=dtype)
            gate_local = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(128, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, 64), 0)

            T.clear(gate_local)
            for k in range(active_k_steps):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    input_shared,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * 128 : (by + 1) * 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(128, 128):
                if i < actual_rows:
                    gate_up_logits[block_start + i, by * 128 + j] = gate_local[i, j]

        # ---- Kernel 2: Up GEMM only ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 128), threads=256) as (bx, by):
            input_shared = T.alloc_shared((128, 64), dtype=dtype)
            weight_shared = T.alloc_shared((128, 64), dtype=dtype)
            up_local = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(128, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, 64), 0)

            T.clear(up_local)
            for k in range(active_k_steps):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    input_shared,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * 128 : (by + 1) * 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(128, 128):
                if i < actual_rows:
                    # 存储到不同位置或单独 workspace
                    # 这里简化，直接写回（会覆盖 gate）
                    pass  # 暂时不写

        # ---- Kernel 3: Down GEMM with routing ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, 128), threads=256) as (bx, by):
            # 由于没有完整的 workspace，这个版本不完整
            pass

    return None  # 这个版本不完整


def run_kernel(*args, **kwargs):
    # 回退到 v226 实现
    from xpuoj_data.probe_v226_kernel1_policy_square import run_kernel as v226_run
    return v226_run(*args, **kwargs)
