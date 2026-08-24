import sys, traceback
steps = []

def step(i, fn):
    try:
        fn()
        steps.append(f'step {i}: OK')
    except Exception as e:
        steps.append(f'step {i}: FAIL: {e}')
        traceback.print_exc()

import torch
step(0, lambda: torch.cuda.set_device(0))
step(1, lambda: torch.cuda.get_device_name(0))
step(2, lambda: torch.cuda.current_device())
step(3, lambda: torch.empty((1,1), device='cpu'))
step(4, lambda: torch.empty((1,1), device='cuda'))
step(5, lambda: torch.zeros((10,10), device='cuda', dtype=torch.float16))
step(6, lambda: torch.randn((10,10), device='cuda'))
x = torch.zeros((10,10), device='cuda', dtype=torch.float16)
step(7, lambda: x.cpu())
step(8, lambda: x.to('cpu'))
for s in steps:
    print(s)