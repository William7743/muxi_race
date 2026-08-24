import os, torch, sys

os.environ['PYTHONPATH'] = '/opt/tilelang-metax-v0.1.10'
os.environ['MACA_PATH'] = '/opt/maca'
os.environ['LD_LIBRARY_PATH'] = '/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu'

print('device_count:', torch.cuda.device_count())
for i in range(4):
    try:
        name = torch.cuda.get_device_name(i)
        print(f'device[{i}]:', name)
    except Exception as e:
        print(f'device[{i}] err:', e)

# try to set device 1
for idx in [0, 1]:
    try:
        torch.cuda.set_device(idx)
        print(f'set_device({idx}) OK')
        x = torch.zeros((1,), device=f'cuda:{idx}')
        print(f'alloc cuda:{idx} OK')
        break
    except Exception as e:
        print(f'set_device({idx}) FAIL:', e)