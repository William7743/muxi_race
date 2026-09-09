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
  void* workspace = ((void*)((char*)buf_dyn_shmem + 0));
  void* workspace_1 = ((void*)((char*)buf_dyn_shmem + 0));
  half_t Ks[16];
  float acc_s[4];
  float mx[1];
  float sm[1];
  half_t acc_cast[4];
  half_t Vwide[16];
  uint Vpacked[8];
  half_t Vmma[16];
  half_t Vs[16];
  float acc_o[16];
  half_t Os_local_cast[4];
  int i_s = (BI[((((int)blockIdx.y) * 4096) + ((int)blockIdx.x))] * 16);
  if (i_s <= ((int)blockIdx.x)) {
    if ((0 <= i_s) && (i_s <= 4080)) {
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        *(uint2*)(Ks + (i * 4)) = *(uint2*)(K + (((((((int)blockIdx.y) * 262144) + (i_s * 64)) + ((((int)threadIdx.x) & 15) * 64)) + (i * 16)) + ((((int)threadIdx.x) >> 4) * 4)));
      }
    } else {
      #pragma unroll
      for (int i_1 = 0; i_1 < 4; ++i_1) {
        half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
        uint2 condval;
        if ((((i_s + (((int)threadIdx.x) & 15)) < 4096) && (0 <= (i_s + (((int)threadIdx.x) & 15))))) {
          condval = *(uint2*)(K + (((((((int64_t)((int)blockIdx.y)) * (int64_t)262144) + (((int64_t)i_s) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)64)) + (((int64_t)i_1) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)4)));
        } else {
          condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
        }
        *(uint2*)(Ks + (i_1 * 4)) = condval;
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      float condval_1;
      if ((((((((int)threadIdx.x) >> 4) * 4) + i_s) + i_2) <= ((int)blockIdx.x))) {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_1 = -MACART_INF_F;
      }
      acc_s[i_2] = condval_1;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      *(uint4*)(((half_t*)Qs) + (((((i_3 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = *(uint4*)(Q + ((((((int)blockIdx.y) * 4194304) + (((int)blockIdx.x) * 1024)) + (i_3 * 512)) + (((int)threadIdx.x) * 8)));
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
    mx[0] = -MACART_INF_F;
    #pragma unroll
    for (int rv = 0; rv < 4; ++rv) {
      mx[0] = max(mx[0], acc_s[rv]);
    }
    mx[0] = tl::AllReduce<tl::MaxOp, 64, 16, 0>::run(mx[0], (&(((float*)workspace)[0])));
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      acc_s[i_4] = exp2f(((acc_s[i_4] - mx[0]) * 0x1.7154764ee6c2fp-3f/*1.803369e-01*/));
    }
    sm[0] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
      sm[0] = (sm[0] + acc_s[rv_1]);
    }
    sm[0] = tl::AllReduce<tl::SumOp, 64, 16, 0>::run(sm[0], (&(((float*)workspace_1)[0])));
    uint2 __1;
    float4 v_ = *(float4*)(acc_s + 0);
    ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
    ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
    *(uint2*)(acc_cast + 0) = __1;
    #pragma unroll
    for (int cb = 0; cb < 4; ++cb) {
      half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_2;
      if (((0 <= ((((int)threadIdx.x) >> 2) + i_s)) && (((((int)threadIdx.x) >> 2) + i_s) < 4096))) {
        condval_2 = *(uint2*)(V + (((((((int64_t)((int)blockIdx.y)) * (int64_t)262144) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)2) * (int64_t)64)) + (((int64_t)i_s) * (int64_t)64)) + (((int64_t)cb) * (int64_t)16)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)4)));
      } else {
        condval_2 = make_uint2(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
      }
      *(uint2*)(Vwide + (cb * 4)) = condval_2;
    }
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
      uint1 v__1 = *(uint1*)(Vwide + (p * 2));
      Vpacked[p] = (*(uint *)(&(v__1)));
    }
    #pragma unroll
    for (int cb_1 = 0; cb_1 < 4; ++cb_1) {
      #pragma unroll
      for (int r = 0; r < 4; ++r) {
        uint lo = __shfl_sync((uint64_t)4294967295, Vpacked[(cb_1 * 2)], ((((((int)threadIdx.x) >> 4) * 16) + (r * 4)) + ((((int)threadIdx.x) & 15) >> 2)), 64);
        uint hi = __shfl_sync((uint64_t)4294967295, Vpacked[((cb_1 * 2) + 1)], ((((((int)threadIdx.x) >> 4) * 16) + (r * 4)) + ((((int)threadIdx.x) & 15) >> 2)), 64);
        uint condval_3;
        if (((((int)threadIdx.x) & 3) < 2)) {
          condval_3 = lo;
        } else {
          condval_3 = hi;
        }
        uint word = condval_3;
        uint condval_4;
        if (((((int)threadIdx.x) & 3) < 2)) {
          condval_4 = lo;
        } else {
          condval_4 = hi;
        }
        ushort bits = ((ushort)(condval_4 >> ((uint)((((int)threadIdx.x) & 1) * 16))));
        uint condval_5;
        if (((((int)threadIdx.x) & 3) < 2)) {
          condval_5 = lo;
        } else {
          condval_5 = hi;
        }
        ushort v__2 = (ushort)(condval_5 >> ((uint)((((int)threadIdx.x) & 1) * 16)));
        Vmma[((cb_1 * 4) + r)] = (*(half_t *)(&(v__2)));
      }
    }
    #pragma unroll
    for (int i_5 = 0; i_5 < 16; ++i_5) {
      Vs[i_5] = Vmma[i_5];
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_6 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    __syncwarp((uint64_t)18446744073709551615);
    for (int j = 0; j < 4; ++j) {
      {
      *(((float32x4*)acc_o) + j) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)Vs) + j),
                    *(((float16x4*)acc_cast) + 0),
                    *(((float32x4*)acc_o) + j));
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
      float4 v__3 = *(float4*)(acc_o + (i_8 * 4));
      ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__3))[0]);
      ((half2*)(&__2))[1] = __float22half2_rn(((float2*)(&v__3))[1]);
      *(uint2*)(Os_local_cast + 0) = __2;
      *(uint2*)(((half_t*)Os) + ((((((int)threadIdx.x) & 15) * 64) + (i_8 * 16)) + ((((int)threadIdx.x) >> 4) * 4))) = *(uint2*)(Os_local_cast + 0);
    }
    __syncwarp((uint64_t)18446744073709551615);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      *(uint4*)(Output + ((((((int)blockIdx.y) * 4194304) + (((int)blockIdx.x) * 1024)) + (i_9 * 512)) + (((int)threadIdx.x) * 8))) = *(uint4*)(((half_t*)Os) + ((i_9 * 512) + (((int)threadIdx.x) * 8)));
    }
  }
}

