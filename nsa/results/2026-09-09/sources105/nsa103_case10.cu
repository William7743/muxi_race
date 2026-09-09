#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void gather_kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) gather_kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* prob = ((void*)((char*)buf_dyn_shmem + 0));
  void* qs = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  void* kv = ((void*)((char*)buf_dyn_shmem + 2048));
  float score[4];
  half_t vp[16];
  float mx[1];
  float sm[1];
  half_t prob_local_cast[4];
  float out[8];
  half_t Output_local_cast_1[4];
  for (int i_s = 0; i_s < 4; ++i_s) {
    int pos = (((BI[((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 6))] * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + i_s);
    float condval;
    if ((((0 <= pos) && (pos < 256)) && (pos <= ((int)blockIdx.x)))) {
      condval = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval = -MACART_INF_F;
    }
    score[i_s] = condval;
  }
  *(uint4*)(((half_t*)qs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((int)blockIdx.x) * 1024) + (((int)threadIdx.x) * 8)));
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    int pos_1 = ((BI[((((int)blockIdx.x) * 2) + i)] * 16) + (((int)threadIdx.x) >> 3));
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_1;
    if ((((0 <= pos_1) && (pos_1 < 256)) && (pos_1 <= ((int)blockIdx.x)))) {
      condval_1 = *(uint4*)(K + ((((int64_t)pos_1) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
    } else {
      condval_1 = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(((half_t*)kv) + (((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
  }
  half_t A_local[4];
  half_t B_local[4];
  __syncthreads();
  for (int ki = 0; ki < 4; ++ki) {
    *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)qs) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)kv) + (((((((((int)threadIdx.x) >> 6) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    {
      *(((float32x4*)score) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)score) + 0));
    };
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    int pos_2 = ((BI[((((int)blockIdx.x) * 2) + i_1)] * 16) + (((int)threadIdx.x) >> 3));
    half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_2;
    if ((((0 <= pos_2) && (pos_2 < 256)) && (pos_2 <= ((int)blockIdx.x)))) {
      condval_2 = *(uint4*)(V + ((((int64_t)pos_2) * (int64_t)64) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
    } else {
      condval_2 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(vp + (i_1 * 8)) = condval_2;
  }
  mx[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 4; ++rv) {
    mx[0] = max(mx[0], score[rv]);
  }
  __syncthreads();
  mx[0] = tl::AllReduce<tl::MaxOp, 128, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    score[i_2] = exp2f(((score[i_2] - mx[0]) * 0x1.71547652b82fep-3f/*1.803369e-01*/));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
    sm[0] = (sm[0] + score[rv_1]);
  }
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 128, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
  uint2 __1;
  float4 __2;
    float4 v_ = *(float4*)(score + 0);
    float4 v__1 = make_float4(sm[0], sm[0], sm[0], sm[0]);
    __2.x = (v_.x/v__1.x);
    __2.y = (v_.y/v__1.y);
    __2.z = (v_.z/v__1.z);
    __2.w = (v_.w/v__1.w);
  ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
  ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
  *(uint2*)(prob_local_cast + 0) = __1;
  __syncthreads();
  *(uint2*)(((half_t*)prob) + (((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(prob_local_cast + 0);
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(uint4*)(((half_t*)kv) + (((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(vp + (i_3 * 8));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(out + (i_4 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
  }
  half_t A_local_1[4];
  half_t B_local_1[8];
  __syncthreads();
  for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
    *(uint2*)(A_local_1 + 0) = *(uint2*)(((half_t*)prob) + (((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
    for (int j = 0; j < 2; ++j) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local_1[((j * 4) + local_id)] = ((half_t*)kv)[(((((((ki_1 * 1024) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((local_id >> 1) + j) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    for (int j_1 = 0; j_1 < 2; ++j_1) {
      {
      *(((float32x4*)out) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_1),
                    *(((float16x4*)A_local_1) + 0),
                    *(((float32x4*)out) + j_1));
    };
    }
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 2; ++i_5) {
    uint2 __3;
    float4 v__2 = *(float4*)(out + (i_5 * 4));
    ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
    ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__2))[1]);
    *(uint2*)(Output_local_cast_1 + 0) = __3;
    *(uint2*)(Output + (((((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + ((((int)threadIdx.x) >> 6) * 32)) + (i_5 * 16)) + (((((int)threadIdx.x) & 63) >> 4) * 4))) = *(uint2*)(Output_local_cast_1 + 0);
  }
}

