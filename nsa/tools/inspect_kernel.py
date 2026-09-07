import importlib.util
import sys
spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('compile begin', flush=True)
kernel = mod._get_kernel(1, 128, 2, 32, 64, 4, 16, 1)
print('compile returned', flush=True)
print(type(kernel), flush=True)
print(kernel.get_kernel_source(), flush=True)
