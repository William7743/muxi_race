"""
Probe v241: Kernel1 reduced accumulator (be=64)
==============================================
基于 v226，尝试减少 kernel1 的 accumulator 数量：
- 将 be1 从 128 改为 64（减少每线程 accumulator 从 128 到 64）
- 目标：降低寄存器压力，提高 occupancy
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
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Kernel 1: Gate/Up fused (be=64) ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, 64), threads=256) as (bx, by):
            input_shared = T.alloc_shared((128, 64), dtype=dtype)
            weight_shared = T.alloc_shared((64, 64), dtype=dtype)
            gate_local = T.alloc_fragment((128, 64), dtype=accum_dtype)
            up_local = T.alloc_fragment((128, 64), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(128, group_size - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, 64), 0)

            T.clear(gate_local)
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
                    gate_w[
                        expert_id,
                        by * 64 : (by + 1) * 64,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_w[
                        expert_id,
                        by * 64 : (by + 1) * 64,
                        k * 64 : (k + 1) * 64,
                    ],
                    weight_shared,
                    coalesced_width=4,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(128, 64):
                if i < actual_rows:
                    up_logits[block_start + i, by * 64 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

        # ---- Kernel 2: Down GEMM ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, 128), threads=256) as (bx, by):
            up_shared = T.alloc_shared((128, 64), dtype=dtype)
            down_shared = T.alloc_shared((128, 64), dtype=dtype)
            out_local = T.alloc_fragment((128, 128), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * 128
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(128, group_size - token_offset))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, 64), 0)

            T.clear(out_local)

            for k in range(active_k_steps):
                T.copy(
                    up_logits[
                        block_start : block_start + 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    up_shared,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * 128 : (by + 1) * 128,
                        k * 64 : (k + 1) * 64,
                    ],
                    down_shared,
                )
                T.gemm(up_shared, down_shared, out_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)
                T.sync_threads()

            for i, j in T.Parallel(128, 128):
                if i < actual_rows:
                    out[block_start + i, by * 128 + j] = (
                        out_local[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
                else:
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
