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
  half_t Ks[8];
  half_t Vpref[8];
  float acc_s[4];
  float mx[1];
  float sm[1];
  half_t acc_cast[4];
  float acc_o[8];
  half_t Os_local_cast[4];
  *(uint4*)(((half_t*)Qs) + ((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((int)blockIdx.x) * 512) + (((int)threadIdx.x) * 8)));
  int i_s = (BI[((int)blockIdx.x)] * 16);
  if (i_s <= ((int)blockIdx.x)) {
    if ((0 <= i_s) && (i_s <= 48)) {
      #pragma unroll
      for (int i = 0; i < 2; ++i) {
        *(uint2*)(Ks + (i * 4)) = *(uint2*)(K + ((((i_s * 32) + ((((int)threadIdx.x) & 15) * 32)) + (i * 16)) + ((((int)threadIdx.x) >> 4) * 4)));
      }
    } else {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
        uint2 condval;
        if ((((i_s + (((int)threadIdx.x) & 15)) < 64) && (0 <= (i_s + (((int)threadIdx.x) & 15))))) {
          condval = *(uint2*)(K + ((((((int64_t)i_s) * (int64_t)32) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)32)) + (((int64_t)i_1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4)));
        } else {
          condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        }
        *(uint2*)(Ks + (i_1 * 4)) = condval;
      }
    }
    if ((0 <= i_s) && (i_s <= 48)) {
      *(uint4*)(Vpref + 0) = *(uint4*)(V + ((i_s * 32) + (((int)threadIdx.x) * 8)));
    } else {
      half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if (((((((int)threadIdx.x) >> 2) + i_s) < 64) && (0 <= ((((int)threadIdx.x) >> 2) + i_s)))) {
        condval_1 = *(uint4*)(V + ((((int64_t)i_s) * (int64_t)32) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(Vpref + 0) = condval_1;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float condval_2;
      if ((((((((int)threadIdx.x) >> 4) * 4) + i_s) + i_2) <= ((int)blockIdx.x))) {
        condval_2 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_2 = -MACART_INF_F;
      }
      acc_s[i_2] = condval_2;
    }
    __syncwarp((uint64_t)18446744073709551615);
    half_t A_local[4];
    for (int ki = 0; ki < 2; ++ki) {
      *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)Qs) + (((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)Ks) + ki),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
    }
    __syncwarp((uint64_t)18446744073709551615);
    mx[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      mx[0] = max(mx[0], acc_s[rv]);
    }
    mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      acc_s[i_3] = exp2f(((acc_s[i_3] - mx[0]) * 0x1.0527dbd5cafffp-2f/*2.550349e-01*/));
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
    *(uint4*)(((half_t*)Vs) + ((((((int)threadIdx.x) >> 2) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Vpref + 0);
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_4 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    __syncwarp((uint64_t)18446744073709551615);
    half_t B_local[8];
    for (int j = 0; j < 2; ++j) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local[((j * 4) + local_id)] = ((half_t*)Vs)[((((((((int)threadIdx.x) >> 4) * 128) + (local_id * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + j) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id >> 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    for (int j_1 = 0; j_1 < 2; ++j_1) {
      {
      *(((float32x4*)acc_o) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + j_1),
                    *(((float16x4*)acc_cast) + 0),
                    *(((float32x4*)acc_o) + j_1));
    };
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      acc_o[i_5] = (acc_o[i_5] / sm[0]);
    }
    __syncwarp((uint64_t)18446744073709551615);
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_o + (i_6 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(Os_local_cast + 0) = __2;
      *(uint2*)(((half_t*)Os) + ((((((int)threadIdx.x) & 15) * 32) + (i_6 * 16)) + ((((int)threadIdx.x) >> 4) * 4))) = *(uint2*)(Os_local_cast + 0);
    }
    __syncwarp((uint64_t)18446744073709551615);
    *(uint4*)(Output + ((((int64_t)((int)blockIdx.x)) * (int64_t)512) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)Os) + (((int)threadIdx.x) * 8));
  }
}
