import sys, os
print('py', sys.executable, sys.version)
print('LD', os.environ.get('LD_LIBRARY_PATH',''))
import torch
print('torch', torch.__version__)
torch.cuda.set_device(0)
print('device ok', torch.cuda.get_device_name(0))
print('device id', torch.cuda.current_device())
import tilelang
print('tilelang ok', tilelang.__version__)