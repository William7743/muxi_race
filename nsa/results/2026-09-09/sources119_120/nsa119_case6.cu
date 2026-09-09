#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const int* __restrict__ BI, const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Ks = ((void*)((char*)buf_dyn_shmem + 0));
  void* acc_cast = ((void*)((char*)buf_dyn_shmem + 0));
  void* reduce_pair = ((void*)((char*)buf_dyn_shmem + 0));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 1024));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 4096));
  float acc_s[4];
  half_t Vpf[16];
  float local_p[4];
  float row_m[1];
  float row_s[1];
  half_t half_p[4];
  float acc_o_sub[8];
  float local_o[8];
  half_t local_h[8];
  int i_s = (BI[((((int)blockIdx.y) * 1024) + ((int)blockIdx.x))] * 32);
  if (i_s <= ((int)blockIdx.x)) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float condval;
      if ((((((((int)threadIdx.x) >> 4) * 4) + i_s) + i) <= ((int)blockIdx.x))) {
        condval = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval = -MACART_INF_F;
      }
      acc_s[i] = condval;
    }
    if ((0 <= i_s) && (i_s <= 992)) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        *(uint4*)(Vpf + (i_1 * 8)) = *(uint4*)(V + (((((((int)blockIdx.y) * 131072) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (i_s * 128)) + ((((int)threadIdx.x) & 7) * 8)));
      }
    } else {
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_1;
        if (((((((((int)threadIdx.x) >> 3) + i_s) >> 4) + i_2) < 64) && (0 <= (((i_2 * 16) + (((int)threadIdx.x) >> 3)) + i_s)))) {
          condval_1 = *(uint4*)(V + (((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)i_2) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
        } else {
          condval_1 = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        }
        *(uint4*)(Vpf + (i_2 * 8)) = condval_1;
      }
    }
    for (int ck_i = 0; ck_i < 2; ++ck_i) {
      *(uint4*)(((half_t*)Qs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.y) * 2097152) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (ck_i * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      #pragma unroll
      for (int i_3 = 0; i_3 < 2; ++i_3) {
        *(uint4*)(((half_t*)Ks) + (((((i_3 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)i_3) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)ck_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      }
      __syncthreads();
      half_t A_local[4];
      half_t B_local[4];
      for (int ki = 0; ki < 4; ++ki) {
        *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)Qs) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)Ks) + (((((((((int)threadIdx.x) >> 6) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
      }
      if (ck_i < 1) {
        __syncthreads();
      }
    }
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      for (int j = 0; j < 32; ++j) {
        local_p[(j & 3)] = acc_s[(j & 3)];
      }
    }
    row_m[0] = max(max(max(-MACART_INF_F, local_p[0]), local_p[1]), max(local_p[2], local_p[3]));
    __syncthreads();
    ((float*)reduce_pair)[((int)threadIdx.x)] = row_m[0];
    __syncthreads();
    row_m[0] = max(row_m[0], ((float*)reduce_pair)[(((int)threadIdx.x) ^ 64)]);
    row_m[0] = max(row_m[0], __shfl_sync((uint64_t)18446744073709551615, row_m[0], ((((int)threadIdx.x) & 63) ^ 32), 64));
    row_m[0] = max(row_m[0], __shfl_sync((uint64_t)18446744073709551615, row_m[0], ((((int)threadIdx.x) & 63) ^ 16), 64));
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      acc_s[i_5] = exp2f(((acc_s[i_5] - row_m[0]) * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
    }
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      for (int j_1 = 0; j_1 < 32; ++j_1) {
        local_p[(j_1 & 3)] = acc_s[(j_1 & 3)];
      }
    }
    row_s[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int j_2 = 0; j_2 < 4; ++j_2) {
      row_s[0] = (row_s[0] + local_p[j_2]);
    }
    ((float*)reduce_pair)[(((int)threadIdx.x) + 128)] = row_s[0];
    __syncthreads();
    row_s[0] = (row_s[0] + ((float*)reduce_pair)[((((int)threadIdx.x) ^ 64) + 128)]);
    row_s[0] = (row_s[0] + __shfl_sync((uint64_t)18446744073709551615, row_s[0], ((((int)threadIdx.x) & 63) ^ 32), 64));
    row_s[0] = (row_s[0] + __shfl_sync((uint64_t)18446744073709551615, row_s[0], ((((int)threadIdx.x) & 63) ^ 16), 64));
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      acc_s[i_7] = (acc_s[i_7] / row_s[0]);
    }
    __syncthreads();
    for (int i_8 = 0; i_8 < 16; ++i_8) {
      for (int j_3 = 0; j_3 < 32; ++j_3) {
        local_p[(j_3 & 3)] = acc_s[(j_3 & 3)];
      }
    }
    #pragma unroll
    for (int j_4 = 0; j_4 < 4; ++j_4) {
      half_p[j_4] = ((half_t)local_p[j_4]);
    }
    *(uint2*)(((half_t*)acc_cast) + (((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(half_p + 0);
    for (int cv_i = 0; cv_i < 2; ++cv_i) {
      if (cv_i == 0) {
        #pragma unroll
        for (int i_9 = 0; i_9 < 2; ++i_9) {
          *(uint4*)(((half_t*)Vs) + (((((i_9 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Vpf + (i_9 * 8));
        }
      } else {
        if ((0 <= i_s) && (i_s <= 992)) {
          #pragma unroll
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            *(uint4*)(((half_t*)Vs) + (((((i_10 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((((int)blockIdx.y) * 131072) + (i_10 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (i_s * 128)) + (cv_i * 64)) + ((((int)threadIdx.x) & 7) * 8)));
          }
        } else {
          #pragma unroll
          for (int i_11 = 0; i_11 < 2; ++i_11) {
            half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_2;
            if (((((((((int)threadIdx.x) >> 3) + i_s) >> 4) + i_11) < 64) && (0 <= (((i_11 * 16) + (((int)threadIdx.x) >> 3)) + i_s)))) {
              condval_2 = *(uint4*)(V + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)i_11) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)cv_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            } else {
              condval_2 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
            }
            *(uint4*)(((half_t*)Vs) + (((((i_11 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_2;
          }
        }
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_o_sub + (i_12 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      __syncthreads();
      half_t A_local_1[4];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        *(uint2*)(A_local_1 + 0) = *(uint2*)(((half_t*)acc_cast) + (((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        for (int j_5 = 0; j_5 < 2; ++j_5) {
          for (int local_id = 0; local_id < 4; ++local_id) {
            B_local_1[((j_5 * 4) + local_id)] = ((half_t*)Vs)[(((((((ki_1 * 1024) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((local_id >> 1) + j_5) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_6 = 0; j_6 < 2; ++j_6) {
          {
      *(((float32x4*)acc_o_sub) + j_6) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_6),
                    *(((float16x4*)A_local_1) + 0),
                    *(((float32x4*)acc_o_sub) + j_6));
    };
        }
      }
      for (int i_13 = 0; i_13 < 16; ++i_13) {
        for (int j_7 = 0; j_7 < 64; ++j_7) {
          local_o[((((j_7 & 31) >> 4) * 4) + (j_7 & 3))] = acc_o_sub[((((j_7 & 31) >> 4) * 4) + (j_7 & 3))];
        }
      }
      #pragma unroll
      for (int j_8 = 0; j_8 < 8; ++j_8) {
        local_h[j_8] = ((half_t)local_o[j_8]);
      }
      __syncwarp((uint64_t)18446744073709551615);
      #pragma unroll
      for (int j_9 = 0; j_9 < 2; ++j_9) {
        *(uint2*)(((half_t*)Vs) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + j_9) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(local_h + (j_9 * 4));
      }
      __syncthreads();
      *(uint4*)(Output + (((((((int)blockIdx.y) * 2097152) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (cv_i * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)Vs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
      if (cv_i < 1) {
        __syncwarp((uint64_t)18446744073709551615);
      }
    }
  }
}
