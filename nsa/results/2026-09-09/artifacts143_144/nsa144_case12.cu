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
  void* Ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 2048));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 2304));
  half_t Qs[16];
  float acc_o[16];
  float ls[1];
  float mx[1];
  uint Vbits[8];
  float acc_s[4];
  half_t Ks_local_cast[8];
  float mx_prev[1];
  float sc[1];
  float sm[1];
  float local_p[4];
  half_t half_p[4];
  half_t acc_cast[4];
  half_t Output_local_cast_1[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint2*)(Qs + (i * 4)) = *(uint2*)(Q + ((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) & 15) * 64)) + (i * 16)) + ((((int)threadIdx.x) >> 4) * 4)) + 1047552) - (((int)blockIdx.y) * 1024)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(acc_o + (i_1 * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  ls[0] = 0x0p+0f/*0.000000e+00*/;
  mx[0] = -MACART_INF_F;
  for (int so = 0; so < 4; ++so) {
    #pragma unroll
    for (int si = 0; si < 2; ++si) {
      int block_id = BI[(((((((int)blockIdx.x) * 8192) + (so * 2)) + si) + 8184) - (((int)blockIdx.y) * 8))];
      if ((0 <= block_id) && (block_id <= ((1023 - ((int)blockIdx.y)) >> 4))) {
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
          uint4 v_ = *(uint4*)(V + ((((((int64_t)((int)blockIdx.x)) * (int64_t)65536) + (((int64_t)block_id) * (int64_t)1024)) + (((int64_t)r) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
          *(uint4*)(Vbits + (r * 4)) = (*(uint4 *)(&(v_)));
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          *(uint4*)(((half_t*)Ks) + (((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((int64_t)((int)blockIdx.x)) * (int64_t)65536) + (((int64_t)block_id) * (int64_t)1024)) + (((int64_t)i_2) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          float condval;
          if ((((((block_id * 16) + ((((int)threadIdx.x) >> 4) * 4)) + i_3) + ((int)blockIdx.y)) <= 1023)) {
            condval = 0x0p+0f/*0.000000e+00*/;
          } else {
            condval = -MACART_INF_F;
          }
          acc_s[i_3] = condval;
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local[4];
        for (int ki = 0; ki < 4; ++ki) {
          *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)Ks) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
          {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)Qs) + ki),
                    *(((float32x4*)acc_s) + 0));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
        #pragma unroll
        for (int r_1 = 0; r_1 < 2; ++r_1) {
          for (int c = 0; c < 4; ++c) {
            ushort2 __1;
            uint2 __2;
              uint2 v__1 = make_uint2(Vbits[((r_1 * 4) + c)], Vbits[((r_1 * 4) + c)]);
              uint2 v__2 = make_uint2(((uint)0)+((uint)16*0), ((uint)0)+((uint)16*1));
              __2.x = (v__1.x >> v__2.x);
              __2.y = (v__1.y >> v__2.y);
            __1.x = (ushort)(__2.x);
            __1.y = (ushort)(__2.y);
            *(uint1*)(Ks_local_cast + (c * 2)) = (*(uint1 *)(&(__1)));
          }
          *(uint4*)(((half_t*)Ks) + (((((r_1 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Ks_local_cast + 0);
        }
        mx_prev[0] = mx[0];
        mx[0] = -MACART_INF_F;
        mx[0] = -MACART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 4; ++rv) {
          mx[0] = max(mx[0], acc_s[rv]);
        }
        mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
        mx[0] = max(mx[0], mx_prev[0]);
        sc[0] = exp2f(((mx_prev[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
        #pragma unroll
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          acc_s[i_4] = exp2f(((acc_s[i_4] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/) - (mx[0] * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/)));
        }
        sm[0] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
          sm[0] = (sm[0] + acc_s[rv_1]);
        }
        sm[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
        ls[0] = ((ls[0] * sc[0]) + sm[0]);
        for (int i_5 = 0; i_5 < 16; ++i_5) {
          for (int j = 0; j < 16; ++j) {
            local_p[(j & 3)] = acc_s[(j & 3)];
          }
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          half_p[j_1] = ((half_t)local_p[j_1]);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 4; ++i_6) {
          acc_cast[i_6] = half_p[i_6];
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 16; ++i_7) {
          acc_o[i_7] = (acc_o[i_7] * sc[0]);
        }
        __syncwarp((uint64_t)18446744073709551615);
        half_t B_local_1[16];
        for (int j_2 = 0; j_2 < 4; ++j_2) {
          for (int local_id = 0; local_id < 4; ++local_id) {
            B_local_1[((j_2 * 4) + local_id)] = ((half_t*)Ks)[(((((((((int)threadIdx.x) >> 4) * 256) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + (j_2 >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_3 = 0; j_3 < 4; ++j_3) {
          {
      *(((float32x4*)acc_o) + j_3) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_3),
                    *(((float16x4*)acc_cast) + 0),
                    *(((float32x4*)acc_o) + j_3));
    };
        }
        __syncwarp((uint64_t)18446744073709551615);
      }
    }
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    acc_o[i_8] = (acc_o[i_8] / ls[0]);
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 4; ++i_9) {
    uint2 __3;
    float4 v__3 = *(float4*)(acc_o + (i_9 * 4));
    ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
    ((half2*)(&__3))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
    *(uint2*)(Output_local_cast_1 + 0) = __3;
    *(uint2*)(Output + ((((((((int)blockIdx.x) * 1048576) + ((((int)threadIdx.x) & 15) * 64)) + (i_9 * 16)) + ((((int)threadIdx.x) >> 4) * 4)) + 1047552) - (((int)blockIdx.y) * 1024))) = *(uint2*)(Output_local_cast_1 + 0);
  }
}
