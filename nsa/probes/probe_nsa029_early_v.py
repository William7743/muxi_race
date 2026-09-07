"""NSA029: load V before QK in simplified S1; ordinary synchronous T.copy."""
import torch
import tilelang
import tilelang.language as T


# ---------------- S=1, D=128, 大 seq_len: chunk kernel ----------------
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_chunk128(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 2  # QK gemm K 维分块
    CV = 4  # PV gemm K 维分块
    DK = D // CK
    DV = D // CV

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BI: T.Tensor(bi_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, batch * head_kv, threads=64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, DK], dtype)
            Ks = T.alloc_shared([BS, DK], dtype)
            Vs = T.alloc_shared([BS, DV], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o_sub = T.alloc_fragment([G, DV], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                # causal mask: 只有首块 (t<BS) 有非零 mask；其余恒 0，但仍需 init acc_s
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                # QK gemm, K 维分 2 块累加（shared 峰值减半）
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # 直接 softmax（S=1 无需 online 增量机制）
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
                # PV gemm, K 维分 4 块（小 acc_o fragment 循环复用）
                for cv_i in T.serial(CV):
                    T.copy(
                        V[i_b, i_s : i_s + BS, i_h, cv_i * DV : (cv_i + 1) * DV],
                        Vs,
                    )
                    T.fill(acc_o_sub, 0)
                    T.gemm(acc_cast, Vs, acc_o_sub, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(
                        acc_o_sub,
                        Output[i_b, i_t, i_h * G : (i_h + 1) * G, cv_i * DV : (cv_i + 1) * DV],
                    )

    return kernel


# ---------------- S=1, D<=64（或 D=128 小 seq_len）: 简化 kernel ----------------
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_simplified(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BI: T.Tensor(bi_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, batch * head_kv, threads=64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, D], dtype)
            Ks = T.alloc_shared([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
                T.fill(acc_o, 0)
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


# ---------------- S>1: 官方 Pipelined + online-softmax 结构 ----------------
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_online(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    num_stages = 2

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BI: T.Tensor(bi_shape, T.int32),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(seq_len, batch * head_kv, threads=64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, D], dtype)
            Ks = T.alloc_shared([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            mx_prev = T.alloc_fragment([G], accum)
            sc = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            ls = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            T.fill(acc_o, 0)
            T.fill(ls, 0)
            T.fill(mx, -T.infinity(accum))
            for s in T.Pipelined(selected_blocks, num_stages=num_stages):
                i_s = BI[i_b, i_t, i_h, s] * BS
                if i_s <= i_t and i_s >= 0:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(
                                i_t >= i_s + j, 0, -T.infinity(acc_s.dtype)
                            )
                    else:
                        T.clear(acc_s)
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(mx, mx_prev)
                    T.fill(mx, -T.infinity(accum))
                    T.reduce_max(acc_s, mx, dim=1, clear=True)
                    for i in T.Parallel(G):
                        sc[i] = T.exp2(mx_prev[i] * scale - mx[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - mx[i] * scale)
                    T.reduce_sum(acc_s, sm, dim=1)
                    for i in T.Parallel(G):
                        ls[i] = ls[i] * sc[i] + sm[i]
                    T.copy(acc_s, acc_cast)
                    for i, j in T.Parallel(G, D):
                        acc_o[i, j] *= sc[i]
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                    T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(G, D):
                acc_o[i, j] /= ls[i]
            T.copy(acc_o, Os)
            T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


_KERNEL_CACHE = {}


def _get_kernel(B, seq_len, H, HQ, D, S, block_size, is_causal):
    """按 shape 缓存；S=1 时按 D/seq_len 选 chunk 或 simplified。"""
    key = (B, seq_len, H, HQ, D, S, block_size, int(is_causal))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        groups = HQ // H
        if S == 1:
            # D=128: chunk 需要足够多的 block 才有 occupancy 收益（grid = seq_len*B*head_kv）
            # B=1 SL=512 (grid=512) chunk 反而慢 (1.02x vs simplified 1.08x)
            if D == 128 and seq_len * B * (HQ // H) >= 1024:
                fn = nsa_chunk128
            else:
                fn = nsa_simplified
        else:
            fn = nsa_online
        kernel = fn(
            batch=B,
            heads=HQ,
            seq_len=seq_len,
            dim=D,
            is_causal=bool(is_causal),
            block_size=block_size,
            groups=groups,
            selected_blocks=S,
        )
        _KERNEL_CACHE[key] = kernel
    return kernel


def run_kernel(
    q,
    k,
    v,
    block_indices,
    output,
    B,
    seq_len,
    H,
    HQ,
    D,
    S,
    block_size,
    is_causal,
):
    if block_indices.dtype != torch.int32:
        block_indices = block_indices.to(torch.int32)
    kernel = _get_kernel(
        int(B),
        int(seq_len),
        int(H),
        int(HQ),
        int(D),
        int(S),
        int(block_size),
        int(is_causal),
    )
    kernel(q, k, v, block_indices, output)


