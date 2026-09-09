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
  void* kv = ((void*)((char*)buf_dyn_shmem + 0));
  void* prob = ((void*)((char*)buf_dyn_shmem + 4096));
  void* qs = ((void*)((char*)buf_dyn_shmem + 4096));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 4096));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 4096));
  void* workspace_2 = ((void*)((char*)buf_dyn_shmem + 4096));
  void* workspace_3 = ((void*)((char*)buf_dyn_shmem + 4096));
  float score[8];
  float mx[1];
  float sm[1];
  half_t prob_local_cast[4];
  float out[4];
  float o_local[4];
  half_t o_half[4];
  half_t o_frag[4];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    for (int vec_s = 0; vec_s < 4; ++vec_s) {
      int pos = (((BI[((((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 4)) + ((((int)threadIdx.x) >> 6) * 2)) + i)] * 16) + (((((int)threadIdx.x) & 63) >> 4) * 4)) + vec_s);
      float condval;
      if ((((0 <= pos) && (pos < 512)) && (pos <= ((int)blockIdx.x)))) {
        condval = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval = -MACART_INF_F;
      }
      score[((i * 4) + vec_s)] = condval;
    }
  }
  for (int kd = 0; kd < 2; ++kd) {
    __syncthreads();
    *(uint2*)(((half_t*)qs) + (((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4))) = *(uint2*)(Q + (((((((int)blockIdx.y) * 524288) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (kd * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      int pos_1 = ((BI[((((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 4)) + (i_1 * 2)) + (((int)threadIdx.x) >> 6))] * 16) + ((((int)threadIdx.x) & 63) >> 2));
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_1;
      if ((((0 <= pos_1) && (pos_1 < 512)) && (pos_1 <= ((int)blockIdx.x)))) {
        condval_1 = *(uint4*)(K + ((((((int64_t)((int)blockIdx.y)) * (int64_t)32768) + (((int64_t)pos_1) * (int64_t)64)) + (((int64_t)kd) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
      } else {
        condval_1 = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
      }
      *(uint4*)(((half_t*)kv) + ((((i_1 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
    }
    half_t A_local[4];
    half_t B_local[8];
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {
      *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)qs) + (((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      for (int j = 0; j < 2; ++j) {
        *(uint2*)(B_local + (j * 4)) = *(uint2*)(((half_t*)kv) + (((((((((int)threadIdx.x) >> 6) * 1024) + (j * 512)) + ((((int)threadIdx.x) & 15) * 32)) + (((((((int)threadIdx.x) & 7) >> 2) + ki) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      }
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        {
      *(((float32x4*)score) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + j_1),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)score) + j_1));
    };
      }
    }
  }
  mx[0] = -MACART_INF_F;
  #pragma unroll
  for (int rv = 0; rv < 8; ++rv) {
    mx[0] = max(mx[0], score[(((rv & 1) * 4) + (rv >> 1))]);
  }
  __syncthreads();
  mx[0] = tl::AllReduce<tl::MaxOp, 128, 64, 0>::run(mx[0], (&(((float*)workspace_3)[0])));
  __syncthreads();
  mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_2)[0])));
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    score[i_2] = exp2f(((score[i_2] - mx[0]) * 0x1.71547652b82fep-3f/*1.803369e-01*/));
  }
  sm[0] = 0x0p+0f/*0.000000e+00*/;
  #pragma unroll
  for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
    sm[0] = (sm[0] + score[(((rv_1 & 1) * 4) + (rv_1 >> 1))]);
  }
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 128, 64, 0>::run(sm[0], (&(((float*)workspace_1)[0])));
  __syncthreads();
  sm[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    uint2 __1;
    float4 __2;
      float4 v_ = *(float4*)(score + (i_3 * 4));
      float4 v__1 = make_float4(sm[0], sm[0], sm[0], sm[0]);
      __2.x = (v_.x/v__1.x);
      __2.y = (v_.y/v__1.y);
      __2.z = (v_.z/v__1.z);
      __2.w = (v_.w/v__1.w);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
    *(uint2*)(prob_local_cast + 0) = __1;
    *(uint2*)(((half_t*)prob) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_3) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(prob_local_cast + 0);
  }
  for (int vd = 0; vd < 2; ++vd) {
    __syncthreads();
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      int pos_2 = ((BI[((((((int)blockIdx.y) * 2048) + (((int)blockIdx.x) * 4)) + (i_4 * 2)) + (((int)threadIdx.x) >> 6))] * 16) + ((((int)threadIdx.x) & 63) >> 2));
      half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_2;
      if ((((0 <= pos_2) && (pos_2 < 512)) && (pos_2 <= ((int)blockIdx.x)))) {
        condval_2 = *(uint4*)(V + ((((((int64_t)((int)blockIdx.y)) * (int64_t)32768) + (((int64_t)pos_2) * (int64_t)64)) + (((int64_t)vd) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)8)));
      } else {
        condval_2 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint4*)(((half_t*)kv) + ((((i_4 * 1024) + ((((int)threadIdx.x) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
    }
    float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(out + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    half_t A_local_1[4];
    half_t B_local_1[4];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      *(uint2*)(A_local_1 + 0) = *(uint2*)(((half_t*)prob) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local_1[local_id] = ((half_t*)kv)[((((((ki_1 * 512) + (((((int)threadIdx.x) & 63) >> 4) * 128)) + (local_id * 32)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id >> 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
      {
      *(((float32x4*)out) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + 0),
                    *(((float16x4*)A_local_1) + 0),
                    *(((float32x4*)out) + 0));
    };
    }
    for (int g = 0; g < 16; ++g) {
      for (int d = 0; d < 32; ++d) {
        o_local[(d & 3)] = out[(d & 3)];
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      o_half[i_5] = ((half_t)o_local[i_5]);
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      o_frag[i_6] = o_half[i_6];
    }
    __syncthreads();
    *(uint2*)(((half_t*)kv) + (((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(o_frag + 0);
    __syncthreads();
    *(uint2*)(Output + (((((((int)blockIdx.y) * 524288) + (((int)blockIdx.x) * 1024)) + ((((int)threadIdx.x) >> 3) * 64)) + (vd * 32)) + ((((int)threadIdx.x) & 7) * 4))) = *(uint2*)(((half_t*)kv) + (((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)));
  }
}

