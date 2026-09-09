"""NSA130: NSA128 plus aligned D128 BS32 scheduler block bounds.
No added prefetch, math, tile, synchronization or compiler-pass change. Experimental.
"""
"""NSA128: NSA127 plus peer v342 packed-register S8 V prefetch.
Only nsa_online_direct_output is transplanted. No peer D128 compile flags.
Static candidate only: own GPU validation pending shared-server handoff.
Peer source SHA256: 8aabb78cc1dfbcc847e4b03cc79587d93100c97d7a3a055df149eedc3b58fb6a.
"""
"""NSA127: combine bounded plain loads (grid>=1024) and ordered bounded prefetch.
Parent NSA125; no changes to attention mathematics or synchronization.
Local combination candidate; requires independent validation.
"""
"""NSA125: K-before-Q in two aligned D64 prefetch paths; with bounded addresses.
Parent NSA123; preserves its plain-factory gain. Diagnostic, not OJ verified.
"""
"""NSA123: isolate NSA122 bounded-load change to the plain D64 factory.
Local candidate; not an OJ score. All other dispatch paths retain NSA109.
"""
# NSA109: NSA103 plus independently confirmed S2 scalar probability conversion.
# NSA103: NSA097 plus scalar output conversion for small H2 D64 S1 workloads.
"""NSA097: v318 TileLang math with standard JIT decorators and no custom compile flags. OJ pending."""
import torch
import tilelang
import tilelang.language as T
from tilelang.maca.intrinsics.layout.mma_layout import make_mma_swizzle_layout


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
    CK = 4 if block_size == 32 and batch * seq_len * head_kv >= 2048 else 2  # Experimental shape-only QK split
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
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
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
    reorder = dim == 64 and groups == 16 and block_size == 16 and batch * seq_len * head_kv >= 4096 and seq_len >= 1024 and seq_len % block_size == 0

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
            sm = T.alloc_fragment([G], accum)
            if not reorder:
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                if reorder:
                    T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Vs[i, j] = V[i_b, i_s + i, i_h, j]
                else:
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
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
        with T.Kernel(batch * head_kv, seq_len, threads=64) as (ibh, i_x):
            i_t = seq_len - 1 - i_x
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_fragment([G, D], dtype)
            Ks = T.alloc_shared([BS, D], dtype)
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
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, D):
                            Ks[i, j] = K[i_b, i_s + i, i_h, j]
                    else:
                        T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                    if is_causal:
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.if_then_else(
                                i_t >= i_s + j, 0, -T.infinity(acc_s.dtype)
                            )
                    else:
                        T.clear(acc_s)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.copy(mx, mx_prev)
                    T.fill(mx, -T.infinity(accum))
                    T.reduce_max(acc_s, mx, dim=1, clear=True)
                    for i in T.Parallel(G):
                        mx[i] = T.max(mx[i], mx_prev[i])
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
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, D):
                            Ks[i, j] = V[i_b, i_s + i, i_h, j]
                    else:
                        T.copy(V[i_b, i_s : i_s + BS, i_h, :], Ks)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.gemm(acc_cast, Ks, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
            for i, j in T.Parallel(G, D):
                acc_o[i, j] /= ls[i]
            T.copy(acc_o, Os)
            T.sync_warp(T.uint64(0xffffffffffffffff))
            T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def nsa_small_group(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    H = heads // groups
    G, D, BS = max(16, groups), dim, block_size
    N = selected_blocks * BS
    KD = min(64, D)
    scale = D ** -0.5 * 1.4426950408889634

    @T.prim_func
    def gather_kernel(
        Q: T.Tensor((batch, seq_len, heads, D), T.float16),
        K: T.Tensor((batch, seq_len, H, D), T.float16),
        V: T.Tensor((batch, seq_len, H, D), T.float16),
        BI: T.Tensor((batch, seq_len, H, selected_blocks), T.int32),
        Output: T.Tensor((batch, seq_len, heads, D), T.float16),
    ):
        with T.Kernel(seq_len, batch * H, threads=64) as (t, bh):
            b, h = bh // H, bh % H
            qs = T.alloc_shared((G, KD), T.float16)
            kv = T.alloc_shared((N, KD), T.float16)
            score = T.alloc_fragment((G, N), T.float32)
            prob = T.alloc_fragment((G, N), T.float16)
            out = T.alloc_fragment((G, KD), T.float32)
            mx = T.alloc_fragment((G,), T.float32)
            sm = T.alloc_fragment((G,), T.float32)
            for g, n in T.Parallel(G, N):
                pos = BI[b, t, h, n // BS] * BS + n % BS
                score[g, n] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                              0, -T.infinity(T.float32))
            for kd in T.serial(D // KD):
                for g, d in T.Parallel(G, KD):
                    qs[g, d] = T.if_then_else(g < groups, Q[b, t, h * groups + g, kd * KD + d], 0)
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               K[b, pos, h, kd * KD + d], 0)
                T.gemm(qs, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            T.reduce_max(score, mx, dim=1, clear=True)
            for g, n in T.Parallel(G, N):
                score[g, n] = T.exp2((score[g, n] - mx[g]) * scale)
            T.reduce_sum(score, sm, dim=1)
            for g, n in T.Parallel(G, N):
                prob[g, n] = score[g, n] / sm[g]
            for vd in T.serial(D // KD):
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               V[b, pos, h, vd * KD + d], 0)
                T.clear(out)
                T.gemm(prob, kv, out, policy=T.GemmWarpPolicy.FullRow)
                for g, d in T.Parallel(G, KD):
                    if g < groups:
                        Output[b, t, h * groups + g, vd * KD + d] = out[g, d]
    return gather_kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_qfragment(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            Qs = T.alloc_fragment([G, D], dtype)
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
                T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_shared_chunk128(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
        with T.Kernel(seq_len, batch * head_kv, threads=128 if block_size == 32 else 64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, DK], dtype)
            Ks = T.alloc_shared([BS, DK], dtype)
            Vs = T.alloc_shared([BS, DV], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_shared([G, BS], dtype)
            acc_o_sub = T.alloc_fragment([G, DV], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_shared_simplified(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
        with T.Kernel(seq_len, batch * head_kv, threads=128 if block_size == 32 else 64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, D], dtype)
            Ks = T.alloc_shared([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_shared([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
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
                T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullCol if groups == 32 else T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
    },
)
def nsa_chunk128_safe_off(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 4 if block_size == 32 and batch * seq_len * head_kv >= 2048 else 2  # Experimental shape-only QK split
    CV = 1 if block_size == 16 else 2  # PV gemm K 维分块
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
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
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


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
    },
)
def nsa_shared_chunk128_safe_off(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 2  # QK gemm K 维分块
    CV = 2  # PV gemm K 维分块
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
        with T.Kernel(seq_len, batch * head_kv, threads=128 if block_size == 32 else 64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, DK], dtype)
            Ks = T.alloc_shared([BS, DK], dtype)
            Vs = T.alloc_shared([BS, DV], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_shared([G, BS], dtype)
            acc_o_sub = T.alloc_fragment([G, DV], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
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


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def nsa_gather(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    H = heads // groups
    G, D, BS = groups, dim, block_size
    N = selected_blocks * BS
    KD = min(32 if (selected_blocks == 4 or (selected_blocks == 8 and batch * seq_len * H >= 256)) else 64, D)
    prefetch_v = selected_blocks == 2
    stage_output = selected_blocks == 4 and seq_len >= 512
    scale = D ** -0.5 * 1.4426950408889634

    @T.prim_func
    def gather_kernel(
        Q: T.Tensor((batch, seq_len, heads, D), T.float16),
        K: T.Tensor((batch, seq_len, H, D), T.float16),
        V: T.Tensor((batch, seq_len, H, D), T.float16),
        BI: T.Tensor((batch, seq_len, H, selected_blocks), T.int32),
        Output: T.Tensor((batch, seq_len, heads, D), T.float16),
    ):
        with T.Kernel(seq_len, batch * H, threads=128) as (t, bh):
            b, h = bh // H, bh % H
            qs = T.alloc_shared((G, KD), T.float16)
            kv = T.alloc_shared((N, KD), T.float16)
            score = T.alloc_fragment((G, N), T.float32)
            prob = T.alloc_shared((G, N), T.float16)
            out = T.alloc_fragment((G, KD), T.float32)
            vp = T.alloc_fragment((N, KD), T.float16)
            mx = T.alloc_fragment((G,), T.float32)
            sm = T.alloc_fragment((G,), T.float32)
            for g, n in T.Parallel(G, N):
                pos = BI[b, t, h, n // BS] * BS + n % BS
                score[g, n] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                              0, -T.infinity(T.float32))
            for kd in T.serial(D // KD):
                for g, d in T.Parallel(G, KD):
                    qs[g, d] = Q[b, t, h * G + g, kd * KD + d]
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               K[b, pos, h, kd * KD + d], 0)
                T.gemm(qs, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if prefetch_v:
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    vp[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               V[b, pos, h, d], 0)
            T.reduce_max(score, mx, dim=1, clear=True)
            for g, n in T.Parallel(G, N):
                score[g, n] = T.exp2((score[g, n] - mx[g]) * scale)
            T.reduce_sum(score, sm, dim=1)
            for g, n in T.Parallel(G, N):
                prob[g, n] = score[g, n] / sm[g]
            for vd in T.serial(D // KD):
                if prefetch_v and vd == 0:
                    T.copy(vp, kv)
                else:
                    for n, d in T.Parallel(N, KD):
                        pos = BI[b, t, h, n // BS] * BS + n % BS
                        kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                                   V[b, pos, h, vd * KD + d], 0)
                T.clear(out)
                T.gemm(prob, kv, out, policy=T.GemmWarpPolicy.FullRow)
                if stage_output:
                    T.copy(out, kv[0:G, :])
                    T.copy(kv[0:G, :], Output[b, t, h * G:(h + 1) * G, vd * KD:(vd + 1) * KD])
                else:
                    for g, d in T.Parallel(G, KD):
                        Output[b, t, h * G + g, vd * KD + d] = out[g, d]
    return gather_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def nsa_vec16(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            T.annotate_layout({Qs: make_mma_swizzle_layout(Qs, vecSize=16),
                               Ks: make_mma_swizzle_layout(Ks, vecSize=16),
                               Vs: make_mma_swizzle_layout(Vs, vecSize=16)})
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
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
                T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    "tl.disable_safe_memory_legalize": True,
})
def nsa_shared_chunk128_manual_sync(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    prefetch_v = head_kv == 1
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 2  # QK gemm K 维分块
    CV = 2  # PV gemm K 维分块
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
        with T.Kernel(seq_len, batch * head_kv, threads=128 if block_size == 32 else 64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, DK], dtype)
            Ks = T.alloc_shared([BS, DK], dtype)
            Vs = T.alloc_shared([BS, DV], dtype)
            Vpf = T.alloc_fragment([BS, DV], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_shared([G, BS], dtype)
            acc_o_sub = T.alloc_fragment([G, DV], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                if prefetch_v:
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, DV):
                            Vpf[i, j] = V[i_b, i_s + i, i_h, j]
                    else:
                        T.copy(V[i_b, i_s : i_s + BS, i_h, 0:DV], Vpf)
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.sync_threads()
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    if ck_i + 1 < CK:
                        T.sync_threads()
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.sync_threads()
                T.copy(acc_s, acc_cast)
                for cv_i in T.serial(CV):
                    if prefetch_v and cv_i == 0:
                        T.copy(Vpf, Vs)
                    else:
                        if i_s >= 0 and i_s + BS <= seq_len:
                            for i, j in T.Parallel(BS, DV):
                                Vs[i, j] = V[i_b, i_s + i, i_h, cv_i * DV + j]
                        else:
                            T.copy(
                                V[i_b, i_s : i_s + BS, i_h, cv_i * DV : (cv_i + 1) * DV],
                                Vs,
                            )
                    T.fill(acc_o_sub, 0)
                    T.sync_threads()
                    T.gemm(acc_cast, Vs, acc_o_sub, policy=T.GemmWarpPolicy.FullRow)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.copy(acc_o_sub, Vs[0:G, :])
                    T.sync_threads()
                    T.copy(
                        Vs[0:G, :],
                        Output[i_b, i_t, i_h * G : (i_h + 1) * G, cv_i * DV : (cv_i + 1) * DV],
                    )
                    if cv_i + 1 < CV:
                        T.sync_warp(T.uint64(0xffffffffffffffff))

    return kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    "tl.disable_safe_memory_legalize": True,
})
def nsa_d128_scheduler(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    assert dim == 128 and groups == 16 and block_size == 32
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    prefetch_v = head_kv == 1
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 2  # QK gemm K 维分块
    CV = 2  # PV gemm K 维分块
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
        with T.Kernel(seq_len, batch * head_kv, threads=128 if block_size == 32 else 64) as (i_t, ibh):
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            Qs = T.alloc_shared([G, DK], dtype)
            Ks = T.alloc_shared([BS, DK], dtype)
            Vs = T.alloc_shared([BS, DV], dtype)
            Vpf = T.alloc_fragment([BS, DV], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_shared([G, BS], dtype)
            acc_o_sub = T.alloc_fragment([G, DV], accum)
            local_o = T.alloc_local([8], accum)
            local_h = T.alloc_local([8], dtype)
            local_p = T.alloc_local([4], accum)
            half_p = T.alloc_local([4], dtype)
            tx = T.get_thread_binding(0)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            block_id = BI[i_b, i_t, i_h, 0]
            if block_id >= 0 and block_id <= i_t // BS:
                i_s = (block_id % (seq_len // BS)) * BS
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                if prefetch_v:
                    for i, j in T.Parallel(BS, DV):
                        Vpf[i, j] = V[i_b, i_s + i, i_h, j]
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.sync_threads()
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    if ck_i + 1 < CK:
                        T.sync_threads()
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.sync_threads()
                for i, j in T.Parallel(G, BS):
                    local_p[j % 4] = acc_s[i, j]
                for j in T.unroll(4):
                    half_p[j] = local_p[j]
                for j in T.vectorized(4):
                    acc_cast[tx % 16, (tx // 64) * 16 + (tx % 64 // 16) * 4 + j] = half_p[j]
                for cv_i in T.serial(CV):
                    if prefetch_v and cv_i == 0:
                        T.copy(Vpf, Vs)
                    else:
                        for i, j in T.Parallel(BS, DV):
                            Vs[i, j] = V[i_b, i_s + i, i_h, cv_i * DV + j]
                    T.fill(acc_o_sub, 0)
                    T.sync_threads()
                    T.gemm(acc_cast, Vs, acc_o_sub, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(G, DV):
                        local_o[(j % 32) // 16 * 4 + j % 4] = acc_o_sub[i, j]
                    for j in T.unroll(8):
                        local_h[j] = local_o[j]
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    for j in T.unroll(2):
                        for k in T.vectorized(4):
                            Vs[tx % 16, (tx // 64) * 32 + j * 16 + (tx % 64 // 16) * 4 + k] = local_h[j * 4 + k]
                    T.sync_threads()
                    T.copy(
                        Vs[0:G, :],
                        Output[i_b, i_t, i_h * G : (i_h + 1) * G, cv_i * DV : (cv_i + 1) * DV],
                    )
                    if cv_i + 1 < CV:
                        T.sync_warp(T.uint64(0xffffffffffffffff))

    return kernel





@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    "tl.disable_safe_memory_legalize": True})
def nsa_chunk128_warp_sync(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    CK = 4 if block_size == 32 and batch * seq_len * head_kv >= 2048 else 2  # Experimental shape-only QK split
    CV = 1 if block_size == 16 else 2  # PV gemm K 维分块
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
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                for ck_i in T.serial(CK):
                    for i, j in T.Parallel(G, DK):
                        Qs[i, j] = Q[i_b, i_t, i_h * G + i, ck_i * DK + j]
                    for i, j in T.Parallel(BS, DK):
                        Ks[i, j] = K[i_b, i_s + i, i_h, ck_i * DK + j]
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = acc_s[i, j] / sm[i]
                T.copy(acc_s, acc_cast)
                for cv_i in T.serial(CV):
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, DV):
                            Vs[i, j] = V[i_b, i_s + i, i_h, cv_i * DV + j]
                    else:
                        T.copy(
                            V[i_b, i_s : i_s + BS, i_h, cv_i * DV : (cv_i + 1) * DV],
                            Vs,
                        )
                    T.fill(acc_o_sub, 0)
                    T.sync_warp(T.uint64(0xffffffffffffffff))
                    T.gemm(acc_cast, Vs, acc_o_sub, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(
                        acc_o_sub,
                        Output[i_b, i_t, i_h * G : (i_h + 1) * G, cv_i * DV : (cv_i + 1) * DV],
                    )

    return kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_k_fragment(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    reorder = dim == 64 and groups == 16 and block_size == 16 and batch * seq_len * head_kv >= 4096 and seq_len % block_size == 0

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
            Ks = T.alloc_fragment([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            if not reorder:
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                if reorder:
                    T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Vs[i, j] = V[i_b, i_s + i, i_h, j]
                else:
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel



@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_d32_v_prefetch(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    prefetch_before_qk = batch == 1 and head_kv == 1

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
            Ks = T.alloc_fragment([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            Vpref = T.alloc_fragment([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                if prefetch_before_qk:
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, D):
                            Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                    else:
                        T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                if not prefetch_before_qk:
                    if i_s >= 0 and i_s + BS <= seq_len:
                        for i, j in T.Parallel(BS, D):
                            Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                    else:
                        T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                T.copy(Vpref, Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_v_prefetch(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            Vpref = T.alloc_fragment([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                else:
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                T.copy(Vpref, Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_k_fragment_v_prefetch(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            Ks = T.alloc_fragment([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            Vpref = T.alloc_fragment([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                else:
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vpref)
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                T.copy(Vpref, Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel





@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_online_direct_output(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
        with T.Kernel(batch * head_kv, seq_len, threads=64) as (ibh, i_x):
            i_t = seq_len - 1 - i_x
            i_b = ibh // head_kv
            i_h = ibh % head_kv
            lane = T.get_thread_binding(0)
            Vbits = T.alloc_local([8], T.uint32)
            Qs = T.alloc_fragment([G, D], dtype)
            Ks = T.alloc_shared([BS, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            local_p = T.alloc_local([4], accum)
            half_p = T.alloc_local([4], dtype)
            mx = T.alloc_fragment([G], accum)
            mx_prev = T.alloc_fragment([G], accum)
            sc = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            ls = T.alloc_fragment([G], accum)
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            T.fill(acc_o, 0)
            T.fill(ls, 0)
            T.fill(mx, -T.infinity(accum))
            for so in T.serial(T.ceildiv(selected_blocks, 2)):
                for si in T.unroll(2):
                    s = so * 2 + si
                    if s < selected_blocks:
                            i_s = BI[i_b, i_t, i_h, s] * BS
                            if i_s <= i_t and i_s >= 0:
                                safe_s = T.min(T.max(i_s, 0), seq_len - BS)
                                for r in T.unroll(2):
                                    Vbits[T.Ramp(r * 4, 1, 4)] = T.reinterpret(V[i_b, safe_s + lane // 8 + r * 8, i_h, T.Ramp((lane % 8) * 8, 1, 8)], "uint32x4")
                                for i, j in T.Parallel(BS, D):
                                    Ks[i, j] = K[i_b, safe_s + i, i_h, j]
                                if is_causal:
                                    for i, j in T.Parallel(G, BS):
                                        acc_s[i, j] = T.if_then_else(
                                            i_t >= i_s + j, 0, -T.infinity(acc_s.dtype)
                                        )
                                else:
                                    T.clear(acc_s)
                                T.sync_warp(T.uint64(0xffffffffffffffff))
                                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                T.sync_warp(T.uint64(0xffffffffffffffff))
                                for r in T.unroll(2):
                                    for c in T.vectorized(8):
                                        Ks[lane // 8 + r * 8, (lane % 8) * 8 + c] = T.reinterpret(T.Cast(T.uint16, Vbits[r * 4 + c // 2] >> ((c % 2) * 16)), T.float16)
                                T.copy(mx, mx_prev)
                                T.fill(mx, -T.infinity(accum))
                                T.reduce_max(acc_s, mx, dim=1, clear=True)
                                for i in T.Parallel(G):
                                    mx[i] = T.max(mx[i], mx_prev[i])
                                for i in T.Parallel(G):
                                    sc[i] = T.exp2(mx_prev[i] * scale - mx[i] * scale)
                                for i, j in T.Parallel(G, BS):
                                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - mx[i] * scale)
                                T.reduce_sum(acc_s, sm, dim=1)
                                for i in T.Parallel(G):
                                    ls[i] = ls[i] * sc[i] + sm[i]
                                for i, j in T.Parallel(G, BS):
                                    local_p[j % 4] = acc_s[i, j]
                                for j in T.unroll(4):
                                    half_p[j] = local_p[j]
                                for i, j in T.Parallel(G, BS):
                                    acc_cast[i, j] = half_p[j % 4]
                                for i, j in T.Parallel(G, D):
                                    acc_o[i, j] *= sc[i]
                                T.sync_warp(T.uint64(0xffffffffffffffff))
                                T.gemm(acc_cast, Ks, acc_o, policy=T.GemmWarpPolicy.FullRow)
                                T.sync_warp(T.uint64(0xffffffffffffffff))
            for i, j in T.Parallel(G, D):
                acc_o[i, j] /= ls[i]
            T.copy(acc_o, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


_KERNEL_CACHE = {}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_h2_scalar_output(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    reorder = dim == 64 and groups == 16 and block_size == 16 and batch * seq_len * head_kv >= 4096 and seq_len >= 1024 and seq_len % block_size == 0

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
            sm = T.alloc_fragment([G], accum)
            local_o = T.alloc_local([16], accum)
            half_o = T.alloc_local([16], dtype)
            tx = T.get_thread_binding(0)
            if not reorder:
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            i_s = BI[i_b, i_t, i_h, 0] * BS
            if i_s <= i_t:
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Ks[i, j] = K[i_b, i_s + i, i_h, j]
                else:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], Ks)
                if reorder:
                    T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                if i_s >= 0 and i_s + BS <= seq_len:
                    for i, j in T.Parallel(BS, D):
                        Vs[i, j] = V[i_b, i_s + i, i_h, j]
                else:
                    T.copy(V[i_b, i_s : i_s + BS, i_h, :], Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                if D == 64 and BS == 16 and G == 16:
                    for i, j in T.Parallel(G, D):
                        local_o[(j // 16) * 4 + j % 4] = acc_o[i, j]
                    for j in T.unroll(16):
                        half_o[j] = local_o[j]
                    for j in T.unroll(4):
                        for k in T.vectorized(4):
                            Os[tx % 16, j * 16 + (tx // 16) * 4 + k] = half_o[j * 4 + k]
                else:
                    T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def nsa_gather_scalar104(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    H = heads // groups
    G, D, BS = groups, dim, block_size
    N = selected_blocks * BS
    KD = min(32 if (selected_blocks == 4 or (selected_blocks == 8 and batch * seq_len * H >= 256)) else 64, D)
    prefetch_v = selected_blocks == 2
    stage_output = selected_blocks == 4 and seq_len >= 512
    scale = D ** -0.5 * 1.4426950408889634

    @T.prim_func
    def gather_kernel(
        Q: T.Tensor((batch, seq_len, heads, D), T.float16),
        K: T.Tensor((batch, seq_len, H, D), T.float16),
        V: T.Tensor((batch, seq_len, H, D), T.float16),
        BI: T.Tensor((batch, seq_len, H, selected_blocks), T.int32),
        Output: T.Tensor((batch, seq_len, heads, D), T.float16),
    ):
        with T.Kernel(seq_len, batch * H, threads=128) as (t, bh):
            b, h = bh // H, bh % H
            qs = T.alloc_shared((G, KD), T.float16)
            kv = T.alloc_shared((N, KD), T.float16)
            score = T.alloc_fragment((G, N), T.float32)
            prob = T.alloc_shared((G, N), T.float16)
            out = T.alloc_fragment((G, KD), T.float32)
            vp = T.alloc_fragment((N, KD), T.float16)
            p_local = T.alloc_local([N // 8], T.float32)
            p_half = T.alloc_local([N // 8], T.float16)
            p_frag = T.alloc_fragment((G, N), T.float16)
            T.annotate_layout({p_frag: T.Fragment((G, N),
                forward_thread_fn=lambda g, n: (n // (N // 2)) * 64 + ((n % 16) // 4) * 16 + g,
                forward_index_fn=lambda g, n: ((n % (N // 2)) // 16) * 4 + n % 4)})
            mx = T.alloc_fragment((G,), T.float32)
            sm = T.alloc_fragment((G,), T.float32)
            for g, n in T.Parallel(G, N):
                pos = BI[b, t, h, n // BS] * BS + n % BS
                score[g, n] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                              0, -T.infinity(T.float32))
            for kd in T.serial(D // KD):
                for g, d in T.Parallel(G, KD):
                    qs[g, d] = Q[b, t, h * G + g, kd * KD + d]
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               K[b, pos, h, kd * KD + d], 0)
                T.gemm(qs, kv, score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            if prefetch_v:
                for n, d in T.Parallel(N, KD):
                    pos = BI[b, t, h, n // BS] * BS + n % BS
                    vp[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                               V[b, pos, h, d], 0)
            T.reduce_max(score, mx, dim=1, clear=True)
            for g, n in T.Parallel(G, N):
                score[g, n] = T.exp2((score[g, n] - mx[g]) * scale)
            T.reduce_sum(score, sm, dim=1)
            for g, n in T.Parallel(G, N):
                p_local[((n % (N // 2)) // 16) * 4 + n % 4] = score[g, n] / sm[g]
            for i in T.unroll(N // 8):
                p_half[i] = p_local[i]
            for g, n in T.Parallel(G, N):
                p_frag[g, n] = p_half[((n % (N // 2)) // 16) * 4 + n % 4]
            T.copy(p_frag, prob)
            for vd in T.serial(D // KD):
                if prefetch_v and vd == 0:
                    T.copy(vp, kv)
                else:
                    for n, d in T.Parallel(N, KD):
                        pos = BI[b, t, h, n // BS] * BS + n % BS
                        kv[n, d] = T.if_then_else(pos >= 0 and pos < seq_len and pos <= t,
                                                   V[b, pos, h, vd * KD + d], 0)
                T.clear(out)
                T.gemm(prob, kv, out, policy=T.GemmWarpPolicy.FullRow)
                if stage_output:
                    T.copy(out, kv[0:G, :])
                    T.copy(kv[0:G, :], Output[b, t, h * G:(h + 1) * G, vd * KD:(vd + 1) * KD])
                else:
                    for g, d in T.Parallel(G, KD):
                        Output[b, t, h * G + g, vd * KD + d] = out[g, d]
    return gather_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_bounded123(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
    scale = (1.0 / dim) ** 0.5 * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    bi_shape = [batch, seq_len, head_kv, selected_blocks]
    dtype = T.float16
    accum = T.float32
    G, BS, D = groups, block_size, dim
    reorder = dim == 64 and groups == 16 and block_size == 16 and batch * seq_len * head_kv >= 4096 and seq_len >= 1024 and seq_len % block_size == 0

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
            sm = T.alloc_fragment([G], accum)
            if not reorder:
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
            block_id = BI[i_b, i_t, i_h, 0]
            if block_id >= 0 and block_id <= i_t // BS:
                i_s = (block_id % (seq_len // BS)) * BS
                for i, j in T.Parallel(BS, D):
                    Ks[i, j] = K[i_b, i_s + i, i_h, j]
                if reorder:
                    T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                for i, j in T.Parallel(BS, D):
                    Vs[i, j] = V[i_b, i_s + i, i_h, j]
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_v_prefetch_ordered125(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            Vpref = T.alloc_fragment([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            block_id = BI[i_b, i_t, i_h, 0]
            if block_id >= 0 and block_id <= i_t // BS:
                i_s = (block_id % (seq_len // BS)) * BS
                for i, j in T.Parallel(BS, D):
                    Ks[i, j] = K[i_b, i_s + i, i_h, j]
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                for i, j in T.Parallel(BS, D):
                    Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                T.copy(Vpref, Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def nsa_simplified_k_fragment_v_prefetch_ordered125(batch, heads, seq_len, dim, is_causal, block_size, groups, selected_blocks):
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
            Ks = T.alloc_fragment([BS, D], dtype)
            Vs = T.alloc_shared([BS, D], dtype)
            Vpref = T.alloc_fragment([BS, D], dtype)
            Os = T.alloc_shared([G, D], dtype)
            acc_s = T.alloc_fragment([G, BS], accum)
            acc_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, D], accum)
            mx = T.alloc_fragment([G], accum)
            sm = T.alloc_fragment([G], accum)
            block_id = BI[i_b, i_t, i_h, 0]
            if block_id >= 0 and block_id <= i_t // BS:
                i_s = (block_id % (seq_len // BS)) * BS
                for i, j in T.Parallel(BS, D):
                    Ks[i, j] = K[i_b, i_s + i, i_h, j]
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Qs)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.if_then_else(i_t >= i_s + j, 0, -T.infinity(accum))
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(Qs, Ks, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                for i, j in T.Parallel(BS, D):
                    Vpref[i, j] = V[i_b, i_s + i, i_h, j]
                T.reduce_max(acc_s, mx, dim=1, clear=True)
                for i, j in T.Parallel(G, BS):
                    acc_s[i, j] = T.exp2((acc_s[i, j] - mx[i]) * scale)
                T.reduce_sum(acc_s, sm, dim=1)
                T.copy(acc_s, acc_cast)
                T.copy(Vpref, Vs)
                T.fill(acc_o, 0)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.gemm(acc_cast, Vs, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(G, D):
                    acc_o[i, j] = acc_o[i, j] / sm[i]
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(acc_o, Os)
                T.sync_warp(T.uint64(0xffffffffffffffff))
                T.copy(Os, Output[i_b, i_t, i_h * G : (i_h + 1) * G, :])

    return kernel


def _get_kernel(B, seq_len, H, HQ, D, S, block_size, is_causal):
    """按 shape 缓存；S=1 时按 D/seq_len 选 chunk 或 simplified。"""
    key = (B, seq_len, H, HQ, D, S, block_size, int(is_causal))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        groups = HQ // H
        if groups < 16:
            fn = nsa_small_group
        elif S == 1 and D == 32 and B * seq_len * H >= 4096:
            fn = nsa_vec16
        elif S == 1 and block_size == 32 and D == 64 and B * seq_len * HQ >= 32768:
            fn = nsa_shared_simplified
        elif S == 1 and block_size == 32 and D == 128 and B * seq_len * H >= 2048:
            fn = nsa_shared_chunk128
        elif S == 1:
            if D == 128 and seq_len * B * (HQ // H) >= 1024:
                fn = nsa_chunk128
            elif D == 64 and block_size == 32 and B * seq_len * HQ >= 32768:
                fn = nsa_qfragment
            else:
                fn = nsa_simplified
        elif (S in (2, 4) or (S == 8 and B * seq_len * H < 512)) and D == 64 and block_size == 16 and groups == 16:
            fn = nsa_gather
        else:
            fn = nsa_online
        if S == 1 and D == 128 and seq_len % block_size == 0 and groups >= 16:
            if fn is nsa_chunk128:
                fn = nsa_chunk128_safe_off
            elif fn is nsa_shared_chunk128:
                fn = nsa_shared_chunk128_safe_off
        if fn is nsa_shared_chunk128_safe_off and block_size == 32 and groups == 16:
            fn = nsa_d128_scheduler if H == 1 else nsa_shared_chunk128_manual_sync
        if fn is nsa_chunk128_safe_off and block_size == 16 and groups == 16:
            fn = nsa_chunk128_warp_sync
        if fn is nsa_simplified and (D == 32 or B * seq_len * H >= 8192):
            fn = nsa_simplified_k_fragment
        if (fn is nsa_online and S == 8 and H == 1 and groups == 16
                and D == 64 and block_size == 16 and seq_len >= 1024 and seq_len % block_size == 0
                and seq_len % block_size == 0):
            fn = nsa_online_direct_output
        if fn is nsa_simplified_k_fragment and D == 32:
            fn = nsa_d32_v_prefetch
        if B == 1 and H == 1 and D == 64 and block_size == 16:
            if fn is nsa_simplified:
                fn = nsa_simplified_v_prefetch
            elif fn is nsa_simplified_k_fragment:
                fn = nsa_simplified_k_fragment_v_prefetch
        if (fn is nsa_simplified and H == 2 and D == 64 and groups == 16
                and block_size == 16 and B * seq_len * H <= 1024):
            fn = nsa_simplified_h2_scalar_output
        if fn is nsa_gather and S == 2 and D == 64 and groups == 16 and block_size == 16:
            fn = nsa_gather_scalar104
        if fn is nsa_simplified and S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 1024:
            fn = nsa_simplified_bounded123
        if S == 1 and D == 64 and block_size == 16 and groups == 16 and seq_len % block_size == 0 and B * seq_len * H >= 4096:
            if fn is nsa_simplified_v_prefetch:
                fn = nsa_simplified_v_prefetch_ordered125
            elif fn is nsa_simplified_k_fragment_v_prefetch:
                fn = nsa_simplified_k_fragment_v_prefetch_ordered125
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
