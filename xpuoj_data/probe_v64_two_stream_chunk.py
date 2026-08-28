# XPU-OJ v64: v293 + 双流分块跨 kernel 重叠（two-stream chunk pipelining）
#
# 动机：MACA 禁异步拷贝，kernel 内 copy/MMA 串行无解（v197/v239/v254/v276/v278 全闭环）；
# 但 kernel 级并发不需要 async copy——把 M 维切成 C 个 chunk：
#   s1: k1(c0) k1(c1) ...          （一条 stream 顺序）
#   s2: k2(c0) k2(c1) ...          （k2(c) 经 event 依赖 k1(c)，与 k1(c+1) 并发）
# kernel2 的时间（case1 占 41%，v196 探针实测）被藏进 kernel1；预期 total ≈ k1 + k2/C。
# chunk 间内存不相交：up_logits/out 按行分块，权重只读——无真依赖。
# 每 block 数学与 v293 逐位一致（tile/k_pack/swizzle/policy/epilogue 全保留）。
# bx_start/bx_count 为编译期参数 → 每 shape 4 个编译变体（C=2）。
import torch
import tilelang
import tilelang.language as T


_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}
_STREAM_CACHE = {}

CHUNKS = 2  # M 维分块数；C=2: total ≈ k1 + k2/2


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_stage1_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    num_blocks_m,
    bx_start,
    bx_count,
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

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(gate_shape, dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
    ):
        # ---- Stage1: gate/up GEMM + silu(gate)*up -> workspace（本 chunk 的 M 块）----
        # smem: A(bt1*bh1) + gate(be1*bh1) + up(be1*bh1) = (128+256)*64*2B = 48KB
        with T.Kernel(bx_count, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):
            input_shared = T.alloc_shared((bt1, bh1), dtype=dtype)
            weight_shared = T.alloc_shared((be1, bh1), dtype=dtype)
            gate_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)
            up_local = T.alloc_fragment((bt1, be1), dtype=accum_dtype)

            T.use_swizzle(2, order="column")

            mbx = bx + bx_start
            expert_id = group_idx_for_bx[mbx]
            block_start = mbx * bt1
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            actual_rows = T.max(0, T.min(bt1, group_size - (block_start - padded_start)))
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
                if i < actual_rows:
                    up_logits[block_start + i, by * be1 + j] = (
                        up_local[i, j]
                        * (
                            gate_local[i, j]
                            * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                        )
                    )

    return kernel


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_stage2_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    bx_start,
    bx_count,
    bt1,
    bh2,
    be2,
    th2,
    weights_dtype,
):
    dtype = T.float16
    accum_dtype = T.float32

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    down_shape = (num_experts, hidden, intermediate)

    @T.prim_func
    def kernel(
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor((total_valid_tokens,), weights_dtype),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        up_logits: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(input_shape, dtype),
    ):
        # ---- Stage2: down GEMM × routed_weight -> out（本 chunk 的 M 块，padding 行写 0）----
        # smem: A(bt1*be2) + down(bh2*be2) = (128+128)*64*2B = 32KB
        with T.Kernel(bx_count, T.ceildiv(hidden, bh2), threads=th2) as (bx, by):
            up_shared = T.alloc_shared((bt1, be2), dtype=dtype)
            down_shared = T.alloc_shared((bh2, be2), dtype=dtype)
            out_local = T.alloc_fragment((bt1, bh2), dtype=accum_dtype)

            T.use_swizzle(2, order="column")

            mbx = bx + bx_start
            expert_id = group_idx_for_bx[mbx]
            block_start = mbx * bt1
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

    return kernel


def _pick_tiles(intermediate):
    return 128, 64, 128, 256


def _stage_kernels(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    weights_dtype,
):
    """编译（缓存）全部 chunk 变体：返回 [(k1_c, k2_c, bx_start, bx_count), ...]"""
    bt1, bh1, be1, th1 = _pick_tiles(intermediate)
    bh2, be2, th2 = 128, 64, 256
    chunks = min(CHUNKS, num_blocks_m)
    chunk = (num_blocks_m + chunks - 1) // chunks

    variants = []
    for c in range(chunks):
        bx_start = c * chunk
        bx_count = min(chunk, num_blocks_m - bx_start)
        if bx_count <= 0:
            continue
        k1 = _KERNEL_CACHE.get(("k1", hidden, intermediate, num_experts, total_padded_tokens, bx_start, bx_count, bt1, bh1, be1, th1))
        if k1 is None:
            k1 = _moe_stage1_kernel(
                hidden, intermediate, num_experts, total_padded_tokens,
                num_blocks_m, bx_start, bx_count, bt1, bh1, be1, th1,
            )
            _KERNEL_CACHE[("k1", hidden, intermediate, num_experts, total_padded_tokens, bx_start, bx_count, bt1, bh1, be1, th1)] = k1
        k2 = _KERNEL_CACHE.get(("k2", hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, bx_start, bx_count, bt1, bh2, be2, th2, str(weights_dtype)))
        if k2 is None:
            k2 = _moe_stage2_kernel(
                hidden, intermediate, num_experts, total_padded_tokens,
                total_valid_tokens, num_blocks_m, bx_start, bx_count,
                bt1, bh2, be2, th2, weights_dtype,
            )
            _KERNEL_CACHE[("k2", hidden, intermediate, num_experts, total_padded_tokens, total_valid_tokens, bx_start, bx_count, bt1, bh2, be2, th2, str(weights_dtype))] = k2
        variants.append((k1, k2, bx_start, bx_count))
    return variants


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


def _get_streams(chunks):
    entry = _STREAM_CACHE.get(chunks)
    if entry is None:
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        evs = [torch.cuda.Event() for _ in range(chunks)]
        entry = (s1, s2, evs)
        _STREAM_CACHE[chunks] = entry
    return entry


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

    variants = _stage_kernels(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
        weights_dtype,
    )

    if len(variants) == 1:
        # 单 chunk 退化为顺序双 kernel（等价 v293）
        k1, k2, _, _ = variants[0]
        k1(
            stacked_expert_tokens, gate_w, up_w,
            group_sizes, group_padded_offsets, group_idx_for_bx,
            up_logits,
        )
        k2(
            down_w, routed_expert_weights,
            group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx,
            up_logits, out,
        )
        return

    s1, s2, evs = _get_streams(len(variants))
    cur = torch.cuda.current_stream()
    # 输入张量由 cur 上的前序工作产生；s1 全部 launch 以此为序
    s1.wait_stream(cur)

    for c, (k1, _, _, _) in enumerate(variants):
        with torch.cuda.stream(s1):
            k1(
                stacked_expert_tokens, gate_w, up_w,
                group_sizes, group_padded_offsets, group_idx_for_bx,
                up_logits,
            )
        evs[c].record(s1)

    for c, (_, k2, _, _) in enumerate(variants):
        s2.wait_event(evs[c])
        with torch.cuda.stream(s2):
            k2(
                down_w, routed_expert_weights,
                group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx,
                up_logits, out,
            )

    # join：cur 上后续工作（含评测计时终点）等两流全部完成
    cur.wait_stream(s1)
    cur.wait_stream(s2)
