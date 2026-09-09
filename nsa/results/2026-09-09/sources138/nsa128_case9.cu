#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(64, 1) kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Os = ((void*)((char*)buf_dyn_shmem + 0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  half_t Ks[16];
  float acc_s[4];
  half_t Vpref[16];
  float mx[1];
  float sm[1];
  half_t acc_cast[4];
  float acc_o[16];
  half_t Os_local_cast[4];
  int block_id = BI[((int)blockIdx.x)];
  if ((0 <= block_id) && (block_id <= (((int)blockIdx.x) >> 4))) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(uint2*)(Ks + (i * 4)) = *(uint2*)(K + (((((block_id & 511) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (i * 16)) + ((((int)threadIdx.x) >> 4) * 4)));
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      *(uint4*)(((half_t*)Qs) + (((((i_1 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((int)blockIdx.x) * 1024) + (i_1 * 512)) + (((int)threadIdx.x) * 8)));
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float condval;
      if ((((((block_id & 511) * 16) + ((((int)threadIdx.x) >> 4) * 4)) + i_2) <= ((int)blockIdx.x))) {
        condval = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval = -MACART_INF_F;
      }
      acc_s[i_2] = condval;
    }
    __syncwarp((uint64_t)18446744073709551615);
    half_t A_local[4];
    for (int ki = 0; ki < 4; ++ki) {
      *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)Qs) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)Ks) + ki),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    __syncwarp((uint64_t)18446744073709551615);
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(uint4*)(Vpref + (i_3 * 8)) = *(uint4*)(V + ((((block_id & 511) * 1024) + (i_3 * 512)) + (((int)threadIdx.x) * 8)));
    }
    mx[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      mx[0] = max(mx[0], acc_s[rv]);
    }
    mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] - mx[0]) * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/));
    }
    sm[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sm[0] = (sm[0] + acc_s[rv_1]);
    }
    sm[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
    uint2 __1;
    float4 v_ = *(float4*)(acc_s + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(acc_cast + 0) = __1;
    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      *(uint4*)(((half_t*)Vs) + (((((i_5 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Vpref + (i_5 * 8));
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_6 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    __syncwarp((uint64_t)18446744073709551615);
    half_t B_local[16];
    for (int j = 0; j < 4; ++j) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local[((j * 4) + local_id)] = ((half_t*)Vs)[(((((((((int)threadIdx.x) >> 4) * 256) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    for (int j_1 = 0; j_1 < 4; ++j_1) {
      {
      *(((float32x4*)acc_o) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + j_1),
                    *(((float16x4*)acc_cast) + 0),
                    *(((float32x4*)acc_o) + j_1));
    };
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 16; ++i_7) {
      acc_o[i_7] = (acc_o[i_7] / sm[0]);
    }
    __syncwarp((uint64_t)18446744073709551615);
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_o + (i_8 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(Os_local_cast + 0) = __2;
      *(uint2*)(((half_t*)Os) + ((((((int)threadIdx.x) & 15) * 64) + (i_8 * 16)) + ((((int)threadIdx.x) >> 4) * 4))) = *(uint2*)(Os_local_cast + 0);
    }
    __syncwarp((uint64_t)18446744073709551615);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      *(uint4*)(Output + (((((int)blockIdx.x) * 1024) + (i_9 * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(((half_t*)Os) + ((i_9 * 512) + (((int)threadIdx.x) * 8)));
    }
  }
}
