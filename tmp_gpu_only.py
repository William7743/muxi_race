import torch, os
torch.cuda.set_device(0)
print('device ok')
print('LD', os.environ.get('LD_LIBRARY_PATH',''))
print('PY', os.environ.get('PYTHONPATH',''))

try:
    x = torch.zeros((16, 16), dtype=torch.float16, device='cuda')
    print('alloc OK', x.shape)
    print('first elem', x[0,0].item())
except Exception as e:
    print('alloc FAIL', e)