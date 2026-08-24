#!/bin/bash
set -o pipefail
# 服务器环境变量
export PATH=/opt/conda/bin:/opt/anaconda3/bin:$PATH
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "=== python ==="
which python
python --version

echo "=== env key ==="
env | grep -E 'PYTHONPATH|MACA_PATH|LD_LIBRARY_PATH|PATH' | sort

echo "=== gpu list ==="
python -c "import torch; print('torch', torch.__version__); print('device', torch.cuda.get_device_name(0))"

echo "=== tilelang import ==="
python -c "import tilelang, tilelang.language as T; print('tilelang ok', tilelang.__version__)"

echo "=== moe verify case3 ==="
cd /root/moe_contest
python - <<'PY'
import sys, time, torch, math
sys.path.insert(0, '.')
from submission import run_kernel

BLOCK = 128

def gen(d_hidden, d_expert, n_routed_experts, group_sum, seed=1234):
    g = torch.Generator(device='cuda')
    g.manual_seed(seed)
    probs = torch.rand(n_routed_experts, device='cuda', generator=g) + 0.05
    counts = torch.multinomial(probs, group_sum, replacement=True, generator=g).bincount(minlength=n_routed_experts)
    group_sizes = counts.to(torch.int32)
    group_offsets = (torch.cumsum(group_sizes, 0) - group_sizes).to(torch.int32)
    padded_sizes = (torch.ceil(group_sizes.float() / BLOCK).int() * BLOCK).to(torch.int32)
    group_padded_offsets = (torch.cumsum(padded_sizes, 0) - padded_sizes).to(torch.int32)
    padded_total = int(padded_sizes.sum().item())
    m_blocks = padded_total // BLOCK
    gidx = torch.zeros(m_blocks, dtype=torch.int32, device='cuda')
    for bx in range(m_blocks):
        ps = bx * BLOCK
        e = 0
        for ee in range(n_routed_experts):
            if ps >= int(group_padded_offsets[ee].item()):
                e = ee
        gidx[bx] = e
    x = torch.randn((padded_total, d_hidden), device='cuda', dtype=torch.float16, generator=g)
    gate_w = torch.randn((n_routed_experts, d_expert, d_hidden), device='cuda', dtype=torch.float16, generator=g) / math.sqrt(d_expert)
    up_w = torch.randn((n_routed_experts, d_expert, d_hidden), device='cuda', dtype=torch.float16, generator=g) / math.sqrt(d_expert)
    down_w = torch.randn((n_routed_experts, d_hidden, d_expert), device='cuda', dtype=torch.float16, generator=g) / math.sqrt(d_hidden)
    rw = torch.rand((group_sum,), device='cuda', dtype=torch.float16, generator=g).float()
    go = torch.nn.functional.pad(group_offsets, (0,1))
    gpo = torch.nn.functional.pad(group_padded_offsets, (0,1))
    go = torch.cat([group_offsets, torch.tensor([group_sum], dtype=torch.int32, device='cuda')]).to(torch.int32)
    gpo = torch.cat([group_padded_offsets, torch.tensor([padded_total], dtype=torch.int32, device='cuda')]).to(torch.int32)
    out = torch.empty((padded_total, d_hidden), dtype=torch.float16, device='cuda')
    return x, gate_w, up_w, down_w, rw, group_sizes, go, gpo, gidx, out

# 功能 case 2 (小配置)
x2, gw2, uw2, dw2, rw2, gs2, go2, gpo2, gidx2, out2 = gen(3584, 1024, 4, 4544, seed=2)
run_kernel(x2, gw2, uw2, dw2, rw2, gs2, go2, gpo2, gidx2, out2)
torch.cuda.synchronize()
print('case2 kernel ok')

# 功能 case 1 (大配置)
x1, gw1, uw1, dw1, rw1, gs1, go1, gpo1, gidx1, out1 = gen(7168, 2048, 8, 2272, seed=1)
run_kernel(x1, gw1, uw1, dw1, rw1, gs1, go1, gpo1, gidx1, out1)
torch.cuda.synchronize()
print('case1 kernel ok')

# 性能 case 3
x3, gw3, uw3, dw3, rw3, gs3, go3, gpo3, gidx3, out3 = gen(7168, 2048, 64, 9088, seed=3)
run_kernel(x3, gw3, uw3, dw3, rw3, gs3, go3, gpo3, gidx3, out3)
torch.cuda.synchronize()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(10):
    run_kernel(x3, gw3, uw3, dw3, rw3, gs3, go3, gpo3, gidx3, out3)
torch.cuda.synchronize()
dt = (time.time() - t0) / 10 * 1000
print(f'case3 avg {dt:.3f} ms')
PY