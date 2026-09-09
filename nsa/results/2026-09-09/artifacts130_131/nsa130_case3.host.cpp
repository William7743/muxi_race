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
  void* Vs = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  void* Qs = ((void*)((char*)buf_dyn_shmem + 2048));
  float acc_s[4];
  float mx[1];
  float sm[1];
  half_t acc_cast[4];
  float acc_o_sub[32];
  half_t Output_local_cast[4];
  int i_s = (BI[((int)blockIdx.x)] * 16);
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
    for (int ck_i = 0; ck_i < 2; ++ck_i) {
      #pragma unroll
      for (int i_1 = 0; i_1 < 2; ++i_1) {
        *(uint4*)(((half_t*)Qs) + (((((i_1 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + (((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + (((int64_t)i_1) * (int64_t)1024)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)ck_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      }
      #pragma unroll
      for (int i_2 = 0; i_2 < 2; ++i_2) {
        *(uint4*)(((half_t*)Ks) + (((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(K + (((((((int64_t)i_2) * (int64_t)1024) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)128)) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)ck_i) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
      }
      __syncwarp((uint64_t)18446744073709551615);
      half_t A_local[4];
      half_t B_local[4];
      for (int ki = 0; ki < 4; ++ki) {
        *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)Qs) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        *(uint2*)(B_local + 0) = *(uint2*)(((half_t*)Ks) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
        {
      *(((float32x4*)acc_s) + 0) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + 0),
                    *(((float16x4*)A_local) + 0),
                    *(((float32x4*)acc_s) + 0));
    };
      }
      __syncwarp((uint64_t)18446744073709551615);
    }
    mx[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      mx[0] = max(mx[0], acc_s[rv]);
    }
    mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace_1)[0])));
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      acc_s[i_3] = exp2f(((acc_s[i_3] - mx[0]) * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
    }
    sm[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sm[0] = (sm[0] + acc_s[rv_1]);
    }
    sm[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm[0], (&(((float*)workspace)[0])));
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = (acc_s[i_4] / sm[0]);
    }
    uint2 __1;
    float4 v_ = *(float4*)(acc_s + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(acc_cast + 0) = __1;
    if ((0 <= i_s) && (i_s <= 240)) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 4; ++i_5) {
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_5 * 256)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(V + (((i_5 * 512) + (i_s * 128)) + (((int)threadIdx.x) * 8)));
      }
    } else {
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval_1;
        if (((((((((int)threadIdx.x) >> 4) + i_s) >> 2) + i_6) < 64) && (0 <= (((i_6 * 4) + (((int)threadIdx.x) >> 4)) + i_s)))) {
          condval_1 = *(uint4*)(V + (((((int64_t)i_6) * (int64_t)512) + (((int64_t)i_s) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          condval_1 = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        }
        *(uint4*)(((half_t*)Vs) + ((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i_6 * 256)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_6 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_1;
      }
    }
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o_sub + (i_7 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    __syncwarp((uint64_t)18446744073709551615);
    half_t B_local_1[32];
    for (int j = 0; j < 8; ++j) {
      for (int local_id = 0; local_id < 4; ++local_id) {
        B_local_1[((j * 4) + local_id)] = ((half_t*)Vs)[((((((((j >> 2) * 1024) + ((((int)threadIdx.x) >> 4) * 256)) + (local_id * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((j & 3) >> 1)) & 1) * 32)) + ((((local_id >> 1) + (j & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (local_id & 1)) & 1) * 8)) + (((int)threadIdx.x) & 7))];
      }
    }
    for (int j_1 = 0; j_1 < 8; ++j_1) {
      {
      *(((float32x4*)acc_o_sub) + j_1) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_1),
                    *(((float16x4*)acc_cast) + 0),
                    *(((float32x4*)acc_o_sub) + j_1));
    };
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      uint2 __2;
      float4 v__1 = *(float4*)(acc_o_sub + (i_8 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__1))[1]);
      *(uint2*)(Output_local_cast + 0) = __2;
      *(uint2*)(Output + ((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)128)) + (((int64_t)i_8) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(Output_local_cast + 0);
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
    
    if (4096 > 65536) {
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size for kernel_kernel to %d", 4096);
        return -1;
    }
    return 0;

    return 0;
}

extern "C" TL_EXPORT int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, int* __restrict__ BI, half_t* __restrict__ Output, mcStream_t stream=mcStreamDefault) {
	kernel_kernel<<<dim3(256, 1, 1), dim3(64, 1, 1), 4096, stream>>>(BI, K, Output, Q, V);
	TILELANG_CHECK_LAST_ERROR("kernel_kernel");

	return 0;
}
