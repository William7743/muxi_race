import sys, traceback
import torch
torch.cuda.set_device(0)
print('device ok')

steps = [
    ('torch.empty cuda',      lambda: torch.empty((10,10), device='cuda', dtype=torch.float16)),
    ('torch.zeros cuda',      lambda: torch.zeros((10,10), device='cuda', dtype=torch.float16)),
    ('torch.randn cuda',      lambda: torch.randn((10,10), device='cuda')),
    ('torch.rand cuda',       lambda: torch.rand((10,10), device='cuda')),
    ('torch.randint cuda',    lambda: torch.randint(0, 100, (10,10), device='cuda')),
    ('torch.empty cpu',       lambda: torch.empty((10,10), device='cpu')),
    ('cpu tensor .to cuda',   lambda: torch.empty((10,10), device='cpu').to('cuda')),
]
for name, fn in steps:
    try:
        t = fn()
        import gc; gc.collect()
        print(f'{name}: OK shape={tuple(t.shape)}')
    except Exception as e:
        print(f'{name}: FAIL {e}')
        traceback.print_exc()