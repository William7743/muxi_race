$ cat /root/moe_contest/v487_s2_direct_emitter.py
] = (
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
        builder = _moe_stage1_prefetch
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
        builder = _moe_stage2_fast
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

