# XPU-OJ v325c: fix cross-expert merge (per-half expert_id) Merged 256-row stage1
# - 4 个独立 shared buffer (xs0/xs1/wts0/wts1)，零共享操作数
# - 每 gemm 后 T.sync_threads()
# - 512 threads, 4 accumulators (128 regs/thread)
# - 权重第二次读从 L2 命中（同一 global 地址）
import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _stage1_merged(
    hidden, intermediate, num_experts, total_padded_tokens, num_blocks_m
):
    scale = 1.44269504
    dtype = T.float16
    accum_dtype = T.float32
    bt = 128
    bh = 64
    be = 128
    n_by = intermediate // be

    @T.prim_func
    def ker(
        stacked_expert_tokens: T.Tensor((total_padded_tokens, hidden), dtype),
        gate_w: T.Tensor((num_experts, intermediate, hidden), dtype),
        up_w: T.Tensor((num_experts, intermediate, hidden), dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor((total_padded_tokens, intermediate), dtype),
    ):
        with T.Kernel((num_blocks_m + 1) // 2, n_by, threads=512) as (bx, by):
            xs0 = T.alloc_shared((bt, bh), dtype=dtype)
            xs1 = T.alloc_shared((bt, bh), dtype=dtype)
            wg0 = T.alloc_shared((be, bh), dtype=dtype)
            wg1 = T.alloc_shared((be, bh), dtype=dtype)
            g0 = T.alloc_fragment((bt, be), dtype=accum_dtype)
            g1 = T.alloc_fragment((bt, be), dtype=accum_dtype)
            u0 = T.alloc_fragment((bt, be), dtype=accum_dtype)
            u1 = T.alloc_fragment((bt, be), dtype=accum_dtype)

            T.use_swizzle(4, order="column")

            b0 = bx * 2
            bs0 = b0 * bt
            expert0 = group_idx_for_bx[b0]
            e1_idx = T.min(b0 + 1, num_blocks_m - 1)
            expert1 = group_idx_for_bx[e1_idx]
            gpstart = group_padded_offsets[expert0]
            gsize = group_sizes[expert_id]
            tok_off = bs0 - gpstart
            act0 = T.max(0, T.min(bt, gsize - tok_off))
            same_e = T.if_then_else(expert0 == expert1, 1, 0)
            act1 = T.max(0, T.min(bt, gsize - tok_off - bt)) * same_e
            active_k = T.if_then_else(act0 > 0, T.ceildiv(hidden, bh), 0)

            T.clear(g0)
            T.clear(g1)
            T.clear(u0)
            T.clear(u1)

            for k in range(active_k):
                T.copy(
                    stacked_expert_tokens[bs0 : bs0 + bt, k * bh : (k + 1) * bh],
                    xs0,
                )
                T.copy(
                    stacked_expert_tokens[bs0 + bt : bs0 + bt * 2, k * bh : (k + 1) * bh],
                    xs1,
                )
                T.copy(
                    gate_w[expert_id, by * be : (by + 1) * be, k * bh : (k + 1) * bh],
                    wg0,
                )
                T.copy(
                    gate_w[expert1, by * be : (by + 1) * be, k * bh : (k + 1) * bh],
                    wg1,
                )
                T.gemm(xs0, wg0, g0, transpose_B=True)
                T.sync_threads()
                T.gemm(xs1, wg1, g1, transpose_B=True)
                T.sync_threads()
                T.copy(
                    up_w[expert_id, by * be : (by + 1) * be, k * bh : (k + 1) * bh],
                    wg0,
                )
                T.copy(
                    up_w[expert1, by * be : (by + 1) * be, k * bh : (k + 1) * bh],
                    wg1,
                )
                T.gemm(xs0, wg0, u0, transpose_B=True)
                T.sync_threads()
                T.gemm(xs1, wg1, u1, transpose_B=True)
                T.sync_threads()

            for i, j in T.Parallel(bt, be):
                if i < act0:
                    g = g0[i, j]
                    up_logits[bs0 + i, by * be + j] = u0[i, j] * (
                        g * (1.0 / (1.0 + T.exp2(-g * scale)))
                    )
            for i, j in T.Parallel(bt, be):
                if i < act1:
                    g = g1[i, j]
                    up_logits[bs0 + bt + i, by * be + j] = u1[i, j] * (
                        g * (1.0 / (1.0 + T.exp2(-g * scale)))
                    )

    return ker


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _stage2_down(
    hidden, intermediate, num_experts,
    total_padded_tokens, total_valid_tokens, num_blocks_m, weights_dtype,
):
    dtype = T.float16
    accum_dtype = T.float32
    bt = 128
    be = 64
    bh = 128

    @T.prim_func
    def ker(
        up_logits: T.Tensor((total_padded_tokens, intermediate), dtype),
        down_w: T.Tensor((num_experts, hidden, intermediate), dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        out: T.Tensor((total_padded_tokens, hidden), dtype),
    ):
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, bh), threads=256) as (bx, by):
            up_s = T.alloc_shared((bt, be), dtype=dtype)
            dn_s = T.alloc_shared((bh, be), dtype=dtype)
            acc = T.alloc_fragment((bt, bh), dtype=accum_dtype)
            T.use_swizzle(4, order="column")
            expert_id = group_idx_for_bx[bx]
            block_start = bx * bt
            gsize = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            gpstart = group_padded_offsets[expert_id]
            tok_off = block_start - gpstart
            actual_rows = T.max(0, T.min(bt, gsize - tok_off))
            active_k = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be), 0)
            T.clear(acc)
            for k in T.Pipelined(active_k, num_stages=1):
                T.copy(up_logits[block_start : block_start + bt, k * be : (k + 1) * be], up_s)
                T.copy(down_w[expert_id, by * bh : (by + 1) * bh, k * be : (k + 1) * be], dn_s)
                T.gemm(up_s, dn_s, acc, transpose_B=True, policy=T.GemmWarpPolicy.Square)
            for i, j in T.Parallel(bt, bh):
                if i < actual_rows:
                    out[block_start + i, by * bh + j] = (
                        acc[i, j] * T.cast(routed_expert_weights[raw_start + tok_off + i], T.float32)
                    )
                else:
                    out[block_start + i, by * bh + j] = 0

    return ker


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


def run_kernel(
    stacked_expert_tokens, gate_w, up_w, down_w,
    routed_expert_weights, group_sizes, group_offsets,
    group_padded_offsets, group_idx_for_bx, out,
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

    ul_key = ("ul", total_padded_tokens, intermediate)
    up_logits = _WORKSPACE_CACHE.get(ul_key)
    if up_logits is None:
        up_logits = torch.empty(
            (total_padded_tokens, intermediate), dtype=torch.float16, device=stacked_expert_tokens.device
        )
        _WORKSPACE_CACHE[ul_key] = up_logits

    k1 = ("s1", hidden, intermediate, num_experts, total_padded_tokens, num_blocks_m)
    f1 = _KERNEL_CACHE.get(k1)
    if f1 is None:
        f1 = _stage1_merged(hidden, intermediate, num_experts, total_padded_tokens, num_blocks_m)
        _KERNEL_CACHE[k1] = f1
    f1(stacked_expert_tokens, gate_w, up_w, group_sizes, group_padded_offsets, group_idx_for_bx, up_logits)

    k2 = ("s2", hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, num_blocks_m, str(weights_dtype))
    f2 = _KERNEL_CACHE.get(k2)
    if f2 is None:
        f2 = _stage2_down(hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, num_blocks_m, weights_dtype)
        _KERNEL_CACHE[k2] = f2
    f2(up_logits, down_w, routed_expert_weights, group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx, out)
