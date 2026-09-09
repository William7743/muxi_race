"""Audit the exact generated workspace ranges and the single-barrier control."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
folder=root/'results/2026-09-09/sources119_120'
a=(folder/'nsa119_case6.cu').read_text()
b=(folder/'nsa120_case6.cu').read_text()
write='    ((float*)reduce_pair)[(((int)threadIdx.x) + 128)] = row_s[0];'
assert a.count(write)==b.count(write)==1
assert b==a.replace(write,'    __syncthreads();\n'+write)
for source in (a,b):
    assert 'void* reduce_pair = ((void*)((char*)buf_dyn_shmem + 0));' in source
    assert 'void* Vs = ((void*)((char*)buf_dyn_shmem + 1024));' in source
    assert 'void* acc_cast = ((void*)((char*)buf_dyn_shmem + 0));' in source
    assert source.count('__shfl_sync((uint64_t)18446744073709551615')==4
    assert 'AllReduce<' not in source
    assert '    __syncthreads();\n    ((float*)reduce_pair)[((int)threadIdx.x)] = row_m[0];\n    __syncthreads();' in source
    assert 'reduce_pair)[(((int)threadIdx.x) ^ 64)]' in source
    assert 'reduce_pair)[((((int)threadIdx.x) ^ 64) + 128)]' in source
    assert write+'\n    __syncthreads();' in source
    tail=source[source.index('acc_s[i_7] = (acc_s[i_7] / row_s[0]);'):]
    assert tail.index('__syncthreads();')<tail.index('*(uint2*)(((half_t*)acc_cast)')
# Explicit physical byte sets: no max-read/sum-write overlap or Vs overlap.
max_bytes=set(range(0,512));sum_bytes=set(range(512,1024));v_bytes=set(range(1024,5120))
assert not max_bytes&sum_bytes and not (max_bytes|sum_bytes)&v_bytes
print(json.dumps(dict(verification='PASS',max_region=[0,512],sum_region=[512,1024],
    cuda_sha256={str(v):hashlib.sha256(s.encode()).hexdigest() for v,s in [(119,a),(120,b)]},
    limitation='Exact case6 generated-source checks, not final ISA/race instrumentation or all-shape proof')))
