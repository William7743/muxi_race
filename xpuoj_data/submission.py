"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM  (v12a (bisect: no merged kernels))

相对 v6 (72.67) 的结构改动（目标：砍 DRAM 流量 + 提高 MMA 效率）：

1. gate/up 拆成单累加器 GEMM kernel。
   v6 一个 block 同时算 gate+up（2 个 fp32 累加器），被迫用 be=64。
   拆开后每个 kernel 只有 1 个 (bt,128) 累加器：
   - th=256/bt=128: 64 regs/thread（与 v6 甜点相同的寄存器预算）
   - th=512/bt=256: 64 regs/thread
   be=128 使 MMA 走满张量核路径，且 x 流量减半 (M*N*K*2/be)。

2. 相邻同专家 128-block 对合并为 256-row block（merged kernel）。
   权重读取次数 = M-block 数：case3 从 ~104 次降到 ~64 次（≈1x 权重流量），
   计算量不变（256 行恰好等于原来两个 128 行 block）。
   合并谓词完全在 kernel 内用 group_idx_for_bx 计算（设备侧，无 host 同步）：
   - pair p=(2i,2i+1) 可合并 <=> 2i+1 < num_blocks_m 且 gidx[2i]==gidx[2i+1]
   - single kernel 处理未被合并覆盖的 block（偶块看后邻、奇块看前邻）
   覆盖性：每个 128-block 要么在某个可合并 pair 里（merged 处理），
   要么由 single kernel 处理，二者互斥且完备。

3. up kernel 的写回循环里就地完成 silu：
   ws 先被 gate kernel 写入 gate 值，up kernel 读 ws（gate）、乘 silu、
   原地写回（逐元素 read-then-write，同线程，无跨块冲突），
   kernel 边界保证串行。省掉独立 silu kernel 与额外 workspace。

4. down kernel 同样做 merged/single 一对，读 ws 做 GEMM，乘 routed weight
   写 out；padding 行显式写 0（与 v6 一致）。

坐标约定与 v6 完全一致：stacked/ws/out 用 padded 坐标，
routed_expert_weights 用 raw 坐标 (raw_start + token_offset + i)。
"""
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
    block_token=128,
    block_n1=128,
    block_k1=64,
    block_n2=128,
    block_k2=64,
    threads_single=256,
    threads_merged=512,
    num_stages=1,
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32

    num_pairs = (num_blocks_m + 1) // 2

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
        routed_expert_weights: T.Tensor(weights_shape, T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        ws: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- G_S: gate GEMM, single 128-row blocks (未被合并覆盖的) ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + block_token,
                        k * block_k1 : (k + 1) * block_k1,
                    ],
                    xs,
                )
                T.copy(
                    gate_w[
                        expert_id,
                        by * block_n1 : (by + 1) * block_n1,
                        k * block_k1 : (k + 1) * block_k1,
                    ],
                    wts,
                )
                T.gemm(xs, wts, acc, transpose_B=True)

            for i, j in T.Parallel(block_token, block_n1):
                if i < actual_rows:
                    ws[block_start + i, by * block_n1 + j] = acc[i, j]

        # ---- U_S: up GEMM + 就地 silu, single 128-row blocks ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                T.copy(
                    stacked_expert_tokens[
                        block_start : block_start + block_token,
                        k * block_k1 : (k + 1) * block_k1,
                    ],
                    xs,
                )
                T.copy(
                    up_w[
                        expert_id,
                        by * block_n1 : (by + 1) * block_n1,
                        k * block_k1 : (k + 1) * block_k1,
                    ],
                    wts,
                )
                T.gemm(xs, wts, acc, transpose_B=True)

            for i, j in T.Parallel(block_token, block_n1):
                if i < actual_rows:
                    ws[block_start + i, by * block_n1 + j] = (
                        ws[block_start + i, by * block_n1 + j]
                        * (1.0 / (1.0 + T.exp2(-ws[block_start + i, by * block_n1 + j] * scale)))
                        * acc[i, j]
                    )

        # ---- D_S: down GEMM, single 128-row blocks ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, block_n2), threads=threads_single) as (bx, by):
            hs = T.alloc_shared((block_token, block_k2), dtype=dtype)
            ds = T.alloc_shared((block_n2, block_k2), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n2), dtype=accum_dtype)

            T.use_swizzle(4)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(intermediate, block_k2), num_stages=num_stages):
                T.copy(
                    ws[
                        block_start : block_start + block_token,
                        k * block_k2 : (k + 1) * block_k2,
                    ],
                    hs,
                )
                T.copy(
                    down_w[
                        expert_id,
                        by * block_n2 : (by + 1) * block_n2,
                        k * block_k2 : (k + 1) * block_k2,
                    ],
                    ds,
                )
                T.gemm(hs, ds, acc, transpose_B=True)

            for i, j in T.Parallel(block_token, block_n2):
                if i < actual_rows:
                    out[block_start + i, by * block_n2 + j] = (
                        acc[i, j] * routed_expert_weights[raw_start + token_offset + i]
                    )
                else:
                    out[block_start + i, by * block_n2 + j] = 0

    return kernel


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
        int(stacked_expert_tokens.device.index or 0),
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
        str(stacked_expert_tokens.dtype),
    )
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = torch.empty(
            (int(stacked_expert_tokens.shape[0]), int(intermediate)),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = ws
    return ws


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

    ws = _get_workspace(stacked_expert_tokens, intermediate)
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
        ws,
        out,
    )
