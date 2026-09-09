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
  void* KV = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 2048));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 2304));
  half_t Qs[16];
  float out[16];
  float ls[1];
  float mx[1];
  float score0[4];
  float score1[4];
  float prev[1];
  float combined[4];
  float sc[1];
  float sm0[1];
  float local_p[4];
  half_t half_p[4];
  half_t prob[4];
  half_t Output_local_cast[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint2*)(Qs + (i * 4)) = *(uint2*)(Q + ((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) & 15) * 64)) + (i * 16)) + ((((int)threadIdx.x) >> 4) * 4)) + 1047552) - (((int)blockIdx.y) * 1024)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(out + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  ls[0] = 0x0p+0f/*0.000000e+00*/;
  mx[0] = -MACART_INF_F;
  for (int pair = 0; pair < 4; ++pair) {
    int s0 = (BI[((((((int)blockIdx.x) * 8192) + (pair * 2)) + 8184) - (((int)blockIdx.y) * 8))] * 16);
    int s1 = (BI[((((((int)blockIdx.x) * 8192) + (pair * 2)) + 8185) - (((int)blockIdx.y) * 8))] * 16);
    if (((0 <= s0) && ((s0 + ((int)blockIdx.y)) <= 1023)) || ((0 <= s1) && ((s1 + ((int)blockIdx.y)) <= 1023))) {
      float broadcast_var_1 = -MACART_INF_F;
      *(float4*)(score0 + 0) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      float broadcast_var_2 = -MACART_INF_F;
      *(float4*)(score1 + 0) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      if ((0 <= s0) && ((s0 + ((int)blockIdx.y)) <= 1023)) {
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          *(uint4*)(((half_t*)KV) + (((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int)blockIdx.x) * 65536) + (i_2 * 512)) + (min(s0, 1008) * 64)) + (((int)threadIdx.x) * 8)));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          float condval;
          if (((((((((int)threadIdx.x) >> 4) * 4) + s0) + i_3) + ((int)blockIdx.y)) <= 1023)) {
            condval = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval = -MACART_INF_F;
          }
          score0[i_3] = condval;
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local[4];
        for (int ki = 0; ki < 4; ++ki) {
          *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)KV) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
          {
      *(((float32x4*)score0) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)Qs) + ki),
                    *(((float32x4*)score0) + 0));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
      }
      if ((0 <= s1) && ((s1 + ((int)blockIdx.y)) <= 1023)) {
        #pragma unroll
        for (int i_4 = 0; i_4 < 2; ++i_4) {
          *(uint4*)(((half_t*)KV) + (((((i_4 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int)blockIdx.x) * 65536) + (i_4 * 512)) + (min(s1, 1008) * 64)) + (((int)threadIdx.x) * 8)));
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 4; ++i_5) {
          float condval_1;
          if (((((((((int)threadIdx.x) >> 4) * 4) + s1) + i_5) + ((int)blockIdx.y)) <= 1023)) {
            condval_1 = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval_1 = -MACART_INF_F;
          }
          score1[i_5] = condval_1;
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local_1[4];
        for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
          *(uint2*)(B_local_1 + 0) = *(uint2*)(((half_t*)KV) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
          {
      *(((float32x4*)score1) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + 0),
                    *(((float16x4*)Qs) + ki_1),
                    *(((float32x4*)score1) + 0));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
      }
      if ((0 <= s0) && ((s0 + ((int)blockIdx.y)) <= 1023)) {
        #pragma unroll
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          *(uint4*)(((half_t*)KV) + (((((i_6 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int)blockIdx.x) * 65536) + (i_6 * 512)) + (min(s0, 1008) * 64)) + (((int)threadIdx.x) * 8)));
        }
      }
      prev[0] = mx[0];
      #pragma unroll
      for (int i_7 = 0; i_7 < 4; ++i_7) {
        combined[i_7] = max(score0[i_7], score1[i_7]);
      }
      mx[0] = -MACART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 4; ++rv) {
        mx[0] = max(mx[0], combined[rv]);
      }
      mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
      mx[0] = max(prev[0], mx[0]);
      sc[0] = exp2f(((prev[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
      #pragma unroll
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        score0[i_8] = exp2f(((score0[i_8] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
        score1[i_8] = exp2f(((score1[i_8] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        combined[i_9] = (score0[i_9] + score1[i_9]);
      }
      sm0[0] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
        sm0[0] = (sm0[0] + combined[rv_1]);
      }
      sm0[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm0[0], (&(((float*)workspace)[0])));
      ls[0] = ((ls[0] * sc[0]) + sm0[0]);
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        out[i_10] = (out[i_10] * sc[0]);
      }
      if ((0 <= s0) && ((s0 + ((int)blockIdx.y)) <= 1023)) {
        for (int g = 0; g < 16; ++g) {
          for (int n = 0; n < 16; ++n) {
            local_p[(n & 3)] = score0[(n & 3)];
          }
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
          half_p[j] = ((half_t)local_p[j]);
        }
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          prob[i_11] = half_p[i_11];
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local_2[16];
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          for (int local_id = 0; local_id < 4; ++local_id) {
            B_local_2[((j_1 * 4) + local_id)] = ((half_t*)KV)[(((((((((int)threadIdx.x) >> 4) * 256) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j_1 >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          {
      *(((float32x4*)out) + j_2) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_2) + j_2),
                    *(((float16x4*)prob) + 0),
                    *(((float32x4*)out) + j_2));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
      }
      if ((0 <= s1) && ((s1 + ((int)blockIdx.y)) <= 1023)) {
        #pragma unroll
        for (int i_12 = 0; i_12 < 2; ++i_12) {
          *(uint4*)(((half_t*)KV) + (((((i_12 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((int)blockIdx.x) * 65536) + (i_12 * 512)) + (min(s1, 1008) * 64)) + (((int)threadIdx.x) * 8)));
        }
        for (int g_1 = 0; g_1 < 16; ++g_1) {
          for (int n_1 = 0; n_1 < 16; ++n_1) {
            local_p[(n_1 & 3)] = score1[(n_1 & 3)];
          }
        }
        #pragma unroll
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          half_p[j_3] = ((half_t)local_p[j_3]);
        }
        #pragma unroll
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          prob[i_13] = half_p[i_13];
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local_3[16];
        for (int j_4 = 0; j_4 < 4; ++j_4) {
          for (int local_id_1 = 0; local_id_1 < 4; ++local_id_1) {
            B_local_3[((j_4 * 4) + local_id_1)] = ((half_t*)KV)[(((((((((int)threadIdx.x) >> 4) * 256) + (local_id_1 * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j_4 >> 1)) & 1) * 32)) + ((((local_id_1 >> 1) + (j_4 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id_1 & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_5 = 0; j_5 < 4; ++j_5) {
          {
      *(((float32x4*)out) + j_5) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_3) + j_5),
                    *(((float16x4*)prob) + 0),
                    *(((float32x4*)out) + j_5));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
      }
    }
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 16; ++i_14) {
    out[i_14] = (out[i_14] / ls[0]);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 4; ++i_15) {
    uint2 __1;
    float4 v_ = *(float4*)(out + (i_15 * 4));
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(Output_local_cast + 0) = __1;
    *(uint2*)(Output + ((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) & 15) * 64)) + (i_15 * 16)) + ((((int)threadIdx.x) >> 4) * 4)) + 1047552) - (((int)blockIdx.y) * 1024))) = *(uint2*)(Output_local_cast + 0);
  }
}

