# XPU-OJ v213 PROBE: kernel1 权重 shared-resident M-loop（流量级优化）
#
# 硬件依据（流量模型）：case3 权重总流量 5.7GB，占绝对主导；当前结构下每个
# 权重 (expert, by) slice 被该 expert 的每个 token-block CTA 各读一次
# （热在 L2，探针 123766 显示 L2 命中仍比驻留慢 13%）。
# 本探针：grid 改为 (num_experts, inter/be1)，每 CTA 内循环该 expert 的全部
# token-block —— 权重 slice 仅加载一次并常驻 shared，被后续 block 复用，
# 权重片 HBM+L2 读流量从 O(#blocks) 降到 O(1)。
# 并发度仍为 E×(inter/128)（case3=1024 CTA），与现行相当。
# 风险：动态循环边界嵌套（expert→block→k）lowering 未验证过；
# swizzle 先行关闭（grid 语义变了，先测正确性与原始计时）。
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
        group_offsets: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Kernel 1: 权重常驻 shared 的 expert-major M-loop ----
        with T.Kernel(num_experts, T.ceildiv(intermediate, be1), threads=th1) as (ex, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            group_size = group_sizes[ex]
            padded_start = group_padded_offsets[ex]
            # 该 expert 占用的 M128 block 数（padded 后）
            nblocks_e = T.ceildiv(group_padded_offsets[ex + 1] - padded_start, bt1)

            for m in range(nblocks_e):
                block_start = padded_start + m * bt1
                actual_rows = T.max(0, T.min(bt1, group_size - m * bt1))
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
                    # 同一 (expert, by, k) 权重片在该 expert 的全部 m-block 间
                    # 复用：每 m 迭代仅 L1/shared 供给，无跨 CTA 重复拉取
                    T.copy(
                        gate_w[
                            ex,
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
                            ex,
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

        # ---- Kernel 2: 保持 v138 原结构 ----
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
