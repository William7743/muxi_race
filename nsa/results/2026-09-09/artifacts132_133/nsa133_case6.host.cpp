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
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  void* Vs = ((void*)((char*)buf_dyn_shmem + 1024));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 4096));
  float acc_s[4];
  uint Vbits[8];
  float mx[1];
  float sm[1];
  float local_p[4];
  half_t half_p[4];
  half_t Vs_local_cast[8];
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
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
      uint4 v_ = *(uint4*)(V + (((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)r) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      *(uint4*)(Vbits + (r * 4)) = (*(uint4 *)(&(v_)));
    }
    for (int ck_i = 0; ck_i < 2; ++ck_i) {
      *(uint4*)(((half_t*)Qs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int)blockIdx.y) * 2097152) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (ck_i * 64)) + ((((int)threadIdx.x) & 7) * 8)));
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        *(uint4*)(((half_t*)Ks) + (((((i_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)i_1) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)ck_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
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
    mx[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      mx[0] = max(mx[0], acc_s[rv]);
    }
    mx[0] = tl::AllReduce<tl::MaxOp, 128, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      acc_s[i_2] = exp2f(((acc_s[i_2] - mx[0]) * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
    }
    sm[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sm[0] = (sm[0] + acc_s[rv_1]);
    }
    sm[0] = tl::AllReduce<tl::SumOp, 128, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      acc_s[i_3] = (acc_s[i_3] / sm[0]);
    }
    __syncthreads();
    for (int i_4 = 0; i_4 < 16; ++i_4) {
      for (int j = 0; j < 32; ++j) {
        local_p[(j & 3)] = acc_s[(j & 3)];
      }
    }
    #pragma unroll
    for (int j_1 = 0; j_1 < 4; ++j_1) {
      half_p[j_1] = ((half_t)local_p[j_1]);
    }
    *(uint2*)(((half_t*)acc_cast) + (((((((int)threadIdx.x) & 15) * 32) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(half_p + 0);
    for (int cv_i = 0; cv_i < 2; ++cv_i) {
      if (cv_i == 0) {
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
            *(uint1*)(Vs_local_cast + (c * 2)) = (*(uint1 *)(&(__1)));
          }
          *(uint4*)(((half_t*)Vs) + (((((r_1 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Vs_local_cast + 0);
        }
      } else {
        if ((0 <= i_s) && (i_s <= 992)) {
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            *(uint4*)(((half_t*)Vs) + (((((i_5 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + ((((((((int)blockIdx.y) * 131072) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (i_s * 128)) + (cv_i * 64)) + ((((int)threadIdx.x) & 7) * 8)));
          }
        } else {
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_1;
            if (((((((((int)threadIdx.x) >> 3) + i_s) >> 4) + i_6) < 64) && (0 <= (((i_6 * 16) + (((int)threadIdx.x) >> 3)) + i_s)))) {
              condval_1 = *(uint4*)(V + ((((((((int64_t)((int)blockIdx.y)) * (int64_t)131072) + (((int64_t)i_6) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)cv_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
            } else {
              condval_1 = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
            }
            *(uint4*)(((half_t*)Vs) + (((((i_6 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
          }
        }
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(acc_o_sub + (i_7 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      __syncthreads();
      half_t A_local_1[4];
      half_t B_local_1[8];
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        *(uint2*)(A_local_1 + 0) = *(uint2*)(((half_t*)acc_cast) + (((((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 7) >> 2) + ki_1) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          for (int local_id = 0; local_id < 4; ++local_id) {
            B_local_1[((j_2 * 4) + local_id)] = ((half_t*)Vs)[(((((((ki_1 * 1024) + (((((int)threadIdx.x) & 63) >> 4) * 256)) + (local_id * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + ((((local_id >> 1) + j_2) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
          }
        }
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          {
      *(((float32x4*)acc_o_sub) + j_3) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_3),
                    *(((float16x4*)A_local_1) + 0),
                    *(((float32x4*)acc_o_sub) + j_3));
    };
        }
      }
      for (int i_8 = 0; i_8 < 16; ++i_8) {
        for (int j_4 = 0; j_4 < 64; ++j_4) {
          local_o[((((j_4 & 31) >> 4) * 4) + (j_4 & 3))] = acc_o_sub[((((j_4 & 31) >> 4) * 4) + (j_4 & 3))];
        }
      }
      #pragma unroll
      for (int j_5 = 0; j_5 < 8; ++j_5) {
        local_h[j_5] = ((half_t)local_o[j_5]);
      }
      __syncwarp((uint64_t)18446744073709551615);
      #pragma unroll
      for (int j_6 = 0; j_6 < 2; ++j_6) {
        *(uint2*)(((half_t*)Vs) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + j_6) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4))) = *(uint2*)(local_h + (j_6 * 4));
      }
      __syncthreads();
      *(uint4*)(Output + (((((((int)blockIdx.y) * 2097152) + (((int)blockIdx.x) * 2048)) + ((((int)threadIdx.x) >> 3) * 128)) + (cv_i * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)Vs) + (((((((int)threadIdx.x) >> 3) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)));
      if (cv_i < 1) {
        __syncwarp((uint64_t)18446744073709551615);
      }
    }
  }
}


#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" TL_EXPORT const char* get_last_error() {
    return error_buf;
}

extern "C" TL_EXPORT int init() {
    error_buf[0] = '\0';
    
    if (6144 > 65536) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size for kernel_kernel to %d", 6144);
        return -1;
    }
    return 0;

    return 0;
}

extern "C" TL_EXPORT int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, int* __restrict__ BI, half_t* __restrict__ Output, mcStream_t stream=mcStreamDefault) {
	kernel_kernel<<<dim3(1024, 8, 1), dim3(128, 1, 1), 6144, stream>>>(BI, K, Output, Q, V);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}
