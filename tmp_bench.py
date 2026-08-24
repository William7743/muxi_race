import sys, time, torch, math
sys.path.insert(0, '/root/moe_contest')
from submission import run_kernel

BLOCK = 128

def cpu_to_cuda(shape, dtype, gen):
    """CPU RNG → GPU tensor，避开 C500 GPU RNG segfault"""
    if dtype == torch.float16:
        return torch.randn(shape, generator=gen).to('cuda').half()
    elif dtype == torch.float32:
        return torch.rand(shape, generator=gen).to('cuda').float()
    else:
        return torch.randint(0, 2, shape, generator=gen).to('cuda').long()

def gen(d_hidden, d_expert, n_routed_experts, group_sum, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    probs = torch.rand(n_routed_experts, generator=g) + 0.05
    counts = torch.multinomial(probs, group_sum, replacement=True, generator=g).bincount(minlength=n_routed_experts)
    group_sizes = counts.to(torch.int32)
    group_offsets = (torch.cumsum(group_sizes, 0) - group_sizes).to(torch.int32)
    padded_sizes = (torch.ceil(group_sizes.float() / BLOCK).int() * BLOCK).to(torch.int32)
    group_padded_offsets = (torch.cumsum(padded_sizes, 0) - padded_sizes).to(torch.int32)
    padded_total = int(padded_sizes.sum().item())
    m_blocks = padded_total // BLOCK
    gidx = torch.zeros(m_blocks, dtype=torch.int32)
    for bx in range(m_blocks):
        ps = bx * BLOCK
        e = 0
        for ee in range(n_routed_experts):
            if ps >= int(group_padded_offsets[ee].item()):
                e = ee
        gidx[bx] = e
    g2 = torch.Generator(); g2.manual_seed(seed + 1)
    x = cpu_to_cuda((padded_total, d_hidden), torch.float16, g2)
    g3 = torch.Generator(); g3.manual_seed(seed + 2)
    gate_w = cpu_to_cuda((n_routed_experts, d_expert, d_hidden), torch.float16, g3) / math.sqrt(d_expert)
    g4 = torch.Generator(); g4.manual_seed(seed + 3)
    up_w = cpu_to_cuda((n_routed_experts, d_expert, d_hidden), torch.float16, g4) / math.sqrt(d_expert)
    g5 = torch.Generator(); g5.manual_seed(seed + 4)
    down_w = cpu_to_cuda((n_routed_experts, d_hidden, d_expert), torch.float16, g5) / math.sqrt(d_hidden)
    g6 = torch.Generator(); g6.manual_seed(seed + 5)
    rw = cpu_to_cuda((group_sum,), torch.float32, g6)
    go = torch.cat([group_offsets, torch.tensor([group_sum], dtype=torch.int32)]).to(torch.int32)
    gpo = torch.cat([group_padded_offsets, torch.tensor([padded_total], dtype=torch.int32)]).to(torch.int32)
    gidx = gidx.to('cuda')
    go = go.to('cuda'); gpo = gpo.to('cuda')
    out = torch.empty((padded_total, d_hidden), dtype=torch.float16, device='cuda')
    return x, gate_w, up_w, down_w, rw, group_sizes, go, gpo, gidx, out

def ref(tensors, out):
    x = tensors['stacked_expert_tokens']
    gate_w, up_w, down_w = tensors['gate_w'], tensors['up_w'], tensors['down_w']
    rw  = tensors['routed_expert_weights']
    gs, go, gpo = tensors['group_sizes'], tensors['group_offsets'], tensors['group_padded_offsets']
    num_experts = int(gs.numel())
    out.zero_()
    x_f32 = x.float()
    for e in range(num_experts):
        vc = int(gs[e].item())
        if vc == 0: continue
        rs = int(go[e].item()); ps = int(gpo[e].item())
        xe = x_f32[ps:ps+vc]
        gl = torch.matmul(xe, gate_w[e].float().transpose(0,1))
        ul = torch.matmul(xe, up_w[e].float().transpose(0,1))
        ha = torch.sigmoid(gl) * gl * ul
        ye = torch.matmul(ha, down_w[e].float().transpose(0,1))
        ye = ye * rw[rs:rs+vc].float().unsqueeze(1)
        out[ps:ps+vc].copy_(ye.half())

cases = {
    1: dict(d_hidden=7168,d_expert=2048,n_routed_experts=8,group_sum=2272),
    2: dict(d_hidden=3584,d_expert=1024,n_routed_experts=4,group_sum=4544),
    3: dict(d_hidden=7168,d_expert=2048,n_routed_experts=64,group_sum=9088),
}

for cid, cfg in cases.items():
    xt, gw, uw, dw, rw, gs, go, gpo, gidx, out = gen(seed=cid*100+7, **cfg)
    ref_out = torch.empty_like(out)
    ref(dict(stacked_expert_tokens=xt,gate_w=gw,up_w=uw,down_w=dw,routed_expert_weights=rw,group_sizes=gs,group_offsets=go,group_padded_offsets=gpo,group_idx_for_bx=gidx), ref_out)
    for _ in range(10):
        run_kernel(xt, gw, uw, dw, rw, gs, go, gpo, gidx, out)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        run_kernel(xt, gw, uw, dw, rw, gs, go, gpo, gidx, out)
    torch.cuda.synchronize()
    dt = (time.time()-t0)/30*1000.0
    diff = (out.float()-ref_out.float()).abs()
    valid = gs > 0
    max_err_valid = diff[valid].max().item() if valid.any() else 0.0
    try:
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
        acc = 'OK'
    except Exception as e:
        acc = 'FAIL'
    print(f'case{cid}: {dt:.3f}ms valid_max_err={max_err_valid:.6f} allclose={acc}')