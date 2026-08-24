import sys, os, torch
print('PYTHONPATH=', os.environ.get('PYTHONPATH',''))
print('LD_LIBRARY_PATH=', os.environ.get('LD_LIBRARY_PATH',''))
print('torch=', torch.__version__)
torch.cuda.set_device(0)
print('device=', torch.cuda.get_device_name(0))
try:
    x = torch.zeros((16, 16), device='cuda', dtype=torch.float16)
    print('alloc_ok', tuple(x.shape), x.device)
    x[0,0] = 1.0
    print('write_ok', x[0,0].item())
except Exception as e:
    print('FAIL', repr(e))
    import traceback; traceback.print_exc()