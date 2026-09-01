import torch, time

def bw(nbytes, iters=20):
    n = nbytes // 2
    a = torch.empty(n, dtype=torch.float16, device="cuda")
    b = torch.empty(n, dtype=torch.float16, device="cuda")
    b.copy_(a); torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        b.copy_(a)
    en.record(); torch.cuda.synchronize()
    dt = st.elapsed_time(en) / iters / 1000
    return 2 * nbytes / dt / 1e12  # read+write TB/s

for mb in (8, 32, 128, 512, 2048, 4096):
    print(f"{mb:5} MB buffer: {bw(mb*2**20):.2f} TB/s", flush=True)
