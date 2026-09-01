# XPU-OJ v326: Grid dimension swap for L2 locality
#
# 变更量：相比 v282 仅交换两个 T.Kernel 的维度顺序和 context var 名。
# 其他所有代码（包括 tile 大小、threads、k_pack、cw8、swizzle）完全不变。
#
# 原理：
#   CUDA 按 blockIdx.x → blockIdx.y 顺序调度 block。
#   v282: T.Kernel(num_blocks_m, by_blocks) → wave = 一个 by-chunk 的所有 M-blocks
#         → x / up_logits 在下一个 wave 被完全冲刷
#   v326: T.Kernel(by_blocks, num_blocks_m) → wave = 一个 M-block 的所有 by-chunks
#         → x (1.8MB) / up_logits (512KB) 在同一 wave 内被 L2 缓存
#
# 流量节约：x 3.05→0.19GB + up_logits 3.05→0.05GB = -5.86GB (-38%)
# 预期：s ≈ 5.28x → ~84 分
import torch
import tilelang
import tilelang.language as T


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
    weights_dtype,
):
    gu_k_pack = 2 if hidden >= 7000 else 1
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32

    bt1 = 128
    bh1 = 64
    be1 = 128
    bh2 = 128
    be2 = 64
    th1 = 256
    th2 = 256

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
        routed_expert_weights: T.Tensor(weights_shape, weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- Kernel 1: gate/up GEMM + silu(gate)*up → up_logits ----
        # GRID SWAP: (n_by1, num_blocks_m) instead of (num_blocks_m, n_by1)
        with T.Kernel(intermediate // be1, num_blocks_m, threads=th1) as (by, bx):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt1
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_sizes[expert_id] - (block_start - padded_start)))
            active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)

            T.clear(gate_local)
            T.clear(up_local)

            for k in range(active_k_steps):
                T.copy(
                    stacked_expert_tokens[block_start : block_start + bt1, k * bh1 : (k + 1) * bh1],
                    input_shared,
                )
                T.copy(
                    gate_w[expert_id, by * be1 : (by + 1) * be1, k * bh1 : (k + 1) * bh1],
                    weight_shared,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, gate_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()
                T.copy(
                    up_w[expert_id, by * be1 : (by + 1) * be1, k * bh1 : (k + 1) * bh1],
                    weight_shared,
                    coalesced_width=8,
                )
                T.gemm(input_shared, weight_shared, up_local, transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=gu_k_pack)
                T.sync_threads()

            for i, j in T.Parallel(bt1, be1):
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j] * (gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale))))
                    )

        # ---- Kernel 2: down GEMM × routed_weight → out ----
        # GRID SWAP: (hidden/bh2, num_blocks_m) instead of (num_blocks_m, hidden/bh2)
        with T.Kernel(T.ceildiv(hidden, bh2), num_blocks_m, threads=th2) as (by, bx):
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
                    up_logits[block_start : block_start + bt1, k * be2 : (k + 1) * be2],
                    up_shared,
                )
                T.copy(
                    down_w[expert_id, by * bh2 : (by + 1) * bh2, k * be2 : (k + 1) * be2],
                    down_shared,
                )
                T.gemm(up_shared, down_shared, out_local, transpose_B=True, policy=T.GemmWarpPolicy.Square)

            for i, j in T.Parallel(bt1, bh2):
                if i < actual_rows:
                    out[block_start + i, by * bh2 + j] = (
                        out_local[i, j] * T.cast(routed_expert_weights[raw_start + token_offset + i], T.float32)
                    )
                else:
                    out[block_start + i, by * bh2 + j] = 0

    return kernel


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


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

    if routed_expert_weights.dtype == torch.float32:
        weights_dtype = T.float32
    else:
        weights_dtype = T.float16

    up_logits = _WORKSPACE_CACHE.get(("ul", total_padded_tokens, intermediate))
    if up_logits is None:
        up_logits = torch.empty(
            (total_padded_tokens, intermediate), dtype=torch.float16, device=stacked_expert_tokens.device
        )
        _WORKSPACE_CACHE[("ul", total_padded_tokens, intermediate)] = up_logits

    mk = ("main", hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, num_blocks_m, str(weights_dtype))
    fn = _KERNEL_CACHE.get(mk)
    if fn is None:
        fn = _moe_forward_kernel(hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, num_blocks_m, weights_dtype)
        _KERNEL_CACHE[mk] = fn

    fn(
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
