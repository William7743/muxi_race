import sys, math
import torch

torch.cuda.set_device(0)
print('gpu ok', torch.cuda.get_device_name(0))

def gen(d_hidden, d_expert, n_routed_experts, group_sum, raw_total, seed):
    BLOCK = 128
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    probs = torch.rand(n_routed_experts, device='cpu', generator=g) + 0.05
    counts = torch.multinomial(probs, group_sum, replacement=True, generator=g).bincount(minlength=n_routed_experts)
    group_sizes = counts.to(torch.int32)
    group_offsets = (torch.cumsum(group_sizes, 0) - group_sizes).to(torch.int32)
    padded_sizes = (torch.ceil(group_sizes.float() / BLOCK).int() * BLOCK).to(torch.int32)
    group_padded_offsets = (torch.cumsum(padded_sizes, 0) - padded_sizes).to(torch.int32)
    padded_total = int(padded_sizes.sum().item())
    m_blocks = padded_total // BLOCK
    gidx = torch.zeros(m_blocks, dtype=torch.int32, device='cpu')
    for bx in range(m_blocks):
        ps = bx * BLOCK
        e = 0
        for ee in range(n_routed_experts):
            if ps >= int(group_padded_offsets[ee].item()): e = ee
        gidx[bx] = e

    def cpu_rnd(shape, dtype, gen):
        if dtype == torch.float16:
            return torch.randn(shape, generator=gen).to('cuda').half()
        return torch.rand(shape, generator=gen).to('cuda').float()

    x   = cpu_rnd((padded_total, d_hidden),            torch.float16, g)
    gw  = cpu_rnd((n_routed_experts, d_expert, d_hidden), torch.float16, g) / math.sqrt(d_expert)
    uw  = cpu_rnd((n_routed_experts, d_expert, d_hidden), torch.float16, g) / math.sqrt(d_expert)
    dw  = cpu_rnd((n_routed_experts, d_hidden,d_expert), torch.float16, g) / math.sqrt(d_hidden)
    rw  = cpu_rnd((raw_total,), torch.float32, g)
    gidx = gidx.to('cuda')
    go   = torch.cat([group_offsets, torch.tensor([group_sum], dtype=torch.int32)]).to('cuda')
    gpo  = torch.cat([group_padded_offsets, torch.tensor([padded_total], dtype=torch.int32)]).to('cuda')
    out  = torch.empty((padded_total, d_hidden), dtype=torch.float16, device='cuda')
    return x, gw, uw, dw, rw, group_sizes, go, gpo, gidx, out

sys.path.insert(0, '/root/moe_contest')
from submission import run_kernel

cases = {
    1: dict(d_hidden=7168, d_expert=2048, n_routed_experts=8,  group_sum=2272,  raw_total=2272),
    2: dict(d_hidden=3584, d_expert=1024, n_routed_experts=4,  group_sum=4544,  raw_total=4544),
    3: dict(d_hidden=7168, d_expert=2048, n_routed_experts=64, group_sum=9088,  raw_total=9088),
}

for cid, cfg in cases.items():
    tensors = gen(**cfg, seed=cid*100+7)
    for _ in range(5):
        run_kernel(**tensors)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(20):
        run_kernel(**tensors)
    t1.record(); t1.synchronize()
    dt = t0.elapsed_time(t1) / 20
    print(f'case{cid}: {dt:.3f} ms')