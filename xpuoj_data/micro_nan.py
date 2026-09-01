import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        "tl.disable_safe_memory_legalize": True,
        "tl.disable_vectorize_256": True,
    }
)
def mini_safeoff(N):
    dtype = T.float16
    accum = T.float32

    @T.prim_func
    def ker(inp: T.Tensor((128,), dtype), outp: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=256) as (bx):
            val = T.alloc_fragment((128,), dtype=accum)
            for i in T.Parallel(128):
                val[i] = T.cast(inp[i], accum)
            for i in T.Parallel(128):
                if i < N:
                    outp[i] = T.cast(val[i] * 2.0, dtype)
                else:
                    outp[i] = 0.0

    return ker


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def mini_safeon(N):
    dtype = T.float16
    accum = T.float32

    @T.prim_func
    def ker(inp: T.Tensor((128,), dtype), outp: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=256) as (bx):
            val = T.alloc_fragment((128,), dtype=accum)
            for i in T.Parallel(128):
                val[i] = T.cast(inp[i], accum)
            for i in T.Parallel(128):
                if i < N:
                    outp[i] = T.cast(val[i] * 2.0, dtype)
                else:
                    outp[i] = 0.0

    return ker


torch.manual_seed(0)
inp = torch.randn(128, dtype=torch.float16)
inp[100:] = float("nan")
for name, fn in (("safeoff", mini_safeoff), ("safeon", mini_safeon)):
    ker = fn(100)
    out = torch.zeros(128, dtype=torch.float16, device="cuda")
    ker(inp.cuda(), out)
    torch.cuda.synchronize()
    tail = out[100:]
    n_nan = int(torch.isnan(tail.float()).sum())
    print(f"{name}: tail NaN count = {n_nan}/28, tail[:4]={tail[:4].tolist()}", flush=True)
