"""
XPUOJ 比赛 #5 题目 1: TileLang 算子优化 - Fused MoE GEMM  (v47: merged manual-MFMA + official MACA swizzle)

== 手工 MMA 合并版（tilelang.intrinsics 路线）==
目标：实现权重读取减半（合并相邻同专家 block 对）并绕开 T.gemm 的
"共享操作数双 gemm miscompile" 与 (256,128)@512 的性能惩罚。

依据 examples/maca/gemm/example_gemm_intrinsics.py（race 分支官方示例）：
- TensorCoreIntrinEmitter (MACA 16x16x16 MFMA, warp_size=64)
- merged tile: 256 行 x 128 列, 8 warps @512th (4x2 warp 网格, 每 warp 64x64)
- chunk=32, num_stages=1: shared = A(256,32)16K + B(128,32)8K = 24K -> 2 blocks/SM
- 手动 Parallel 装载 + T.annotate_layout swizzle (最后一维 32*16bit=512 可 swizzle)
- G_M 用 stmatrix 直写 global (pid_m/pid_n 重载)
- U_M/D_M 用自定义 fragment store 循环（复刻 _warp_stmatrix_global 的索引映射）
  实现 silu 就地变换 / rwv 乘法 + select

权重 tile 每 k-chunk 只从 global 读一次，供整个 256 行 tile 使用 -> 权重遍数减半。
single 类 kernel (G_S/U_S/D_S) 沿用 v22 已验证的 T.gemm + covered 谓词 (@th256)。

合并谓词（设备侧，穷举验证互斥完备）：
pair p=(2i,2i+1) 可合并 <=> 2i+1 < nbm 且 gidx[2i]==gidx[2i+1]
"""
import torch
import tilelang
import tilelang.language as T
# ===== begin inline race intrinsics =====
from tvm import arith, DataType
import tilelang.language as T


def ldmatrix_32x4_to_shared_16x8_layout_a(thread_id, local_id):
    row = thread_id % 16
    col = (thread_id // 16) * 4 + local_id % 4
    return row, col


def ldmatrix_32x4_to_shared_16x8_layout_b(thread_id, local_id):
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = ((thread_id % 16) // 8) * 4 + local_id % 4
    return row, col


def ldmatrix_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = thread_id % 16
    col = 8 * (thread_id // 16) + local_id % 8
    return row, col


def ldmatrix_trans_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (thread_id // 16) + (thread_id % 8)
    col = 8 * ((thread_id % 16) // 8) + local_id % 8
    return row, col


def ldmatrix_32x16_to_shared_16x32_layout_a(thread_id, local_id):
    row = thread_id % 16
    col = local_id + (thread_id // 16) * 16
    return row, col


def ldmatrix_32x16_to_shared_16x32_layout_b(thread_id, local_id):
    row = (thread_id // 16) * 8 + (thread_id % 8)
    col = local_id + 16 * ((thread_id % 16) // 8)
    return row, col


def mma_store_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


def mma_store_32x2_to_shared_8x8_layout_fp64(thread_id, local_id):
    row = thread_id // 4
    col = (thread_id % 4) * 2 + local_id
    return row, col


# sr represents spatial + reduction layout
# the first axis is spatial while the second axis is reduction
# mma.sync matrix A layout, if wanna trans, please apply map_indices
def shared_16x8_to_mma_a_32x4_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 2 * (j // 4) + (i // 8)


def shared_16x8_to_mma_a_32x4_layout_trans(i, j):
    return shared_16x8_to_mma_a_32x4_layout(j, i)


# mma.sync matrix B layout, if wanna trans, please apply map_indices
def shared_16x8_to_mma_b_32x4_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 4)
    return thread_id, 2 * (i // 8) + (j // 4)


def shared_16x8_to_mma_b_32x4_layout_trans(i, j):
    return shared_16x8_to_mma_b_32x4_layout(j, i)


shared_16x8_to_mma_32x4_layout_sr_a = shared_16x8_to_mma_a_32x4_layout
shared_16x8_to_mma_32x4_layout_sr_b = shared_16x8_to_mma_b_32x4_layout
shared_16x8_to_mma_32x4_layout_rs_a = shared_16x8_to_mma_a_32x4_layout_trans
shared_16x8_to_mma_32x4_layout_rs_b = shared_16x8_to_mma_b_32x4_layout_trans


def shared_16x16_to_mma_a_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x16_to_mma_a_32x8_layout_trans(i, j):
    return shared_16x16_to_mma_a_32x8_layout(j, i)


def shared_16x16_to_mma_b_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (i // 8) + (j // 8) * 2 + (j % 2)


def shared_16x16_to_mma_b_32x8_layout_trans(i, j):
    return shared_16x16_to_mma_b_32x8_layout(j, i)


shared_16x16_to_mma_32x8_layout_sr_a = shared_16x16_to_mma_a_32x8_layout
shared_16x16_to_mma_32x8_layout_sr_b = shared_16x16_to_mma_b_32x8_layout
shared_16x16_to_mma_32x8_layout_rs_a = shared_16x16_to_mma_a_32x8_layout_trans
shared_16x16_to_mma_32x8_layout_rs_b = shared_16x16_to_mma_b_32x8_layout_trans


def shared_16x32_to_mma_a_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_mma_a_32x16_layout_trans(i, j):
    return shared_16x32_to_mma_a_32x16_layout(j, i)


def shared_16x32_to_mma_b_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (i // 8) + (j // 16) * 4 + j % 4


def shared_32x16_to_mma_b_32x16_layout_trans(i, j):
    return shared_16x32_to_mma_b_32x16_layout(j, i)


shared_16x32_to_mma_32x16_layout_sr_a = shared_16x32_to_mma_a_32x16_layout
shared_16x32_to_mma_32x16_layout_sr_b = shared_16x32_to_mma_b_32x16_layout
shared_16x32_to_mma_32x16_layout_rs_a = shared_32x16_to_mma_a_32x16_layout_trans
shared_16x32_to_mma_32x16_layout_rs_b = shared_32x16_to_mma_b_32x16_layout_trans


def mma_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col


def mma_load_a_32x4_to_shared_16x8_layout(thread_id, local_id):
    row = 8 * (local_id % 2) + (thread_id // 4)
    col = 4 * (local_id // 2) + (thread_id % 4)
    return row, col


def mma_load_b_32x4_to_shared_16x8_layout(thread_id, local_id):
    row = 8 * (local_id // 2) + (thread_id // 4)
    col = 4 * (local_id % 2) + (thread_id % 4)
    return row, col


def mma_load_a_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id % 8 // 4) + (thread_id // 4)
    col = 16 * (local_id // 8) + (thread_id % 4) * 4 + (local_id % 4)
    return row, col


def mma_load_a_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
            groupID + 8         Otherwise

    col =  (threadID_in_group * 2) + (i & 0x1)          for ai where i <  4
    (threadID_in_group * 2) + (i & 0x1) + 8      for ai where i >= 4
    """
    row = (thread_id // 4) + 8 * (local_id % 4 // 2)
    col = (thread_id % 4) * 2 + (local_id % 2) + 8 * (local_id // 4)
    return row, col


def mma_load_b_32x16_to_shared_16x32_layout(thread_id, local_id):
    row = 8 * (local_id // 8) + (thread_id // 4)
    col = 16 * (local_id % 8 // 4) + (thread_id % 4) * 4 + (local_id % 4)
    return row, col


def mma_load_b_32x8_to_shared_16x16_layout(thread_id, local_id):
    """
    groupID           = %laneid >> 2
    threadID_in_group = %laneid % 4

    row =  (threadID_in_group * 2) + (i & 0x1)           for bi where i <  2
        (threadID_in_group * 2) + (i & 0x1) + 8       for bi where i >= 2

    col = groupID
    """
    col = (thread_id % 4) * 2 + ((local_id % 4) % 2) + ((local_id % 4) // 2) * 8
    row = (thread_id // 4) + 8 * (local_id // 4)
    return row, col


def shared_16x16_to_mma_32x8_smoothlayout(i, j):
    return (i * 2 + j // 8, j % 8)


def shared_16x32_to_mma_32x16_smoothlayout(i, j):
    return (i * 2 + j // 16, j % 16)


def shared_32x16_to_mma_32x16_smoothlayout(i, j):
    return (i * 2 + j // 16, j % 16)


def get_swizzle_layout(row_idx, col_idx, row_size, dtype: DataType | str, swizzle_bytes=None):
    ana = arith.Analyzer()
    if isinstance(dtype, str):
        dtype = DataType(dtype)
    row_bytes = dtype.bits * row_size // 8
    assert row_bytes % 32 == 0, "Row size must be multiple of 32B."
    if swizzle_bytes is None:
        swizzle_bytes = min(128, row_bytes)
    # 128B swizzle
    #   Use 8 * 8 permuted layout
    #   Every number below corresponds to 16B
    #   0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #   0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    #   0  1  2  3  4  5  6  7    ==>    2  3  0  1  6  7  4  5
    #   0  1  2  3  4  5  6  7    ==>    3  2  1  0  7  6  5  4
    #   0  1  2  3  4  5  6  7    ==>    4  5  6  7  0  1  2  3
    #   0  1  2  3  4  5  6  7    ==>    5  4  7  6  1  0  3  2
    #   0  1  2  3  4  5  6  7    ==>    6  7  4  5  2  3  0  1
    #   0  1  2  3  4  5  6  7    ==>    7  6  5  4  3  2  1  0
    # 64B swizzle
    #  Use 8 * 4 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  0  1  2  3    ==>    0  1  2  3  0  1  2  3
    #  0  1  2  3  4  0  1  2  3    ==>    1  0  3  2  1  0  3  2
    #  0  1  2  3  4  0  1  2  3    ==>    2  3  0  1  2  3  0  1
    #  0  1  2  3  4  0  1  2  3    ==>    3  2  1  0  3  2  1  0
    # 32B swizzle
    #  Use 8 * 2 permuted layout
    #  Every number below corresponds to 16B
    #  0  1  2  3  4  5  6  7    ==>    0  1  2  3  4  5  6  7
    #  0  1  2  3  4  5  6  7    ==>    1  0  3  2  5  4  7  6
    elem_per_16B = 128 // dtype.bits
    col_idx_16B = col_idx // elem_per_16B
    col_idx_in_16B = col_idx % elem_per_16B
    new_col_idx_16B = col_idx_16B ^ (row_idx % (swizzle_bytes // 16))
    return row_idx, ana.simplify(new_col_idx_16B * elem_per_16B + col_idx_in_16B)


def make_mma_swizzle_layout(shared_buf, is_smooth: bool = False):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits % 512 == 0
    if is_smooth or (not can_swizzle):
        return T.Layout(shape, lambda *args: args)

    def transform_func(*args):
        i, j = args[-2:]
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [*args[:-2], new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)

from tvm import DataType
from tvm.runtime import convert
from tvm.tir import const
import tilelang.language as T


def shared_16x4_to_local_64x1_layout_A(i, j):
    thread_id = j * 16 + i
    return thread_id, const(0)


def thread_id_shared_access_64x1_to_16x4_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = thread_id // 16
    return i, j


def shared_4x16_to_local_64x1_layout_B(i, j):
    thread_id = i * 16 + j
    return thread_id, const(0)


def thread_id_shared_access_64x1_to_4x16_layout_B(thread_id, local_id):
    i = thread_id // 16
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_C(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


def shared_16x16_to_ldmatrix_64x4_layout(ind):
    i, j = ind[0], ind[1]
    thread_id, local_id = shared_16x16_to_local_64x4_layout_C(i, j)
    return convert([thread_id, local_id])


def thread_id_shared_access_64x4_to_16x16_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 4 + local_id
    return i, j


def shared_16x16_to_local_64x4_layout_A(i, j):
    thread_id = i + 16 * (j // 4)
    local = j % 4
    return thread_id, local


def thread_id_shared_access_64x4_to_16x16_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def shared_16x16_to_local_64x4_layout_B(i, j):
    thread_id = j + (i // 4) * 16
    local = i % 4
    return thread_id, local


shared_16x16_to_local_64x4_layout_m_n = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_k = shared_16x16_to_local_64x4_layout_A
shared_16x16_to_local_64x4_layout_n_m = shared_16x16_to_local_64x4_layout_B
shared_16x16_to_local_64x4_layout_k_n = shared_16x16_to_local_64x4_layout_B


def thread_id_shared_access_64x4_to_16x16_layout_C_m_n(thread_id, local_id):
    i = local_id + (thread_id // 16) * 4
    j = thread_id % 16
    return i, j


def thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 4
    return i, j


def thread_id_shared_access_64x8_to_16x32_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = (thread_id // 16) * 8 + local_id
    return i, j


def shared_16x32_to_local_64x8_layout_A(i, j):
    thread_id = i + 16 * (j // 8)
    local = j % 8
    return thread_id, local


def thread_id_shared_access_64x8_to_16x32_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 8
    j = thread_id % 16
    return i, j


def shared_16x32_to_local_64x8_layout_B(i, j):
    thread_id = j + (i // 8) * 16
    local = i % 8
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_A(thread_id, local_id):
    i = thread_id % 16
    j = local_id + (thread_id // 16) * 16
    return i, j


def shared_16x64_to_local_64x16_layout_A(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def thread_id_shared_access_64x16_to_16x64_layout_B(thread_id, local_id):
    i = local_id + (thread_id // 16) * 16
    j = thread_id % 16
    return i, j


def shared_16x64_to_local_64x16_layout_B(i, j):
    thread_id = i + 16 * (j // 16)
    local = j % 16
    return thread_id, local


def make_mfma_swizzle_layout(shared_buf, vecSize=8):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    numBanks = 32
    bankBitWidth = 32
    SIMDWidth = 16

    innerDimLength = shape[-1]
    typeWidthInBit = DataType(dtype).bits

    elemsPerOneBanksRow = (numBanks * bankBitWidth) // typeWidthInBit
    perPhase = max(1, elemsPerOneBanksRow // innerDimLength)
    maxPhase = min(SIMDWidth // perPhase, innerDimLength // vecSize)

    def transform(row, col):
        phase = (row // perPhase) % maxPhase
        colOffSwizzled = ((col // vecSize) ^ phase) * vecSize
        colOffOrdered = col % vecSize
        colOff = colOffSwizzled + colOffOrdered
        return row, colOff

    return T.Layout(shape, transform)

from tvm import DataType
from typing import Literal



# the original implementation and insight is from the following code snippet
# 3rdparty/tvm/python/tvm/tir/tensor_intrin/cuda.py#get_ldmatrix_intrin
def get_ldmatrix_offset(
    matrix: Literal["A", "B"],
    row_idx,
    col_idx,
    stride,
    dtype: Literal["float16", "int8", "int4"] = "float16",
    transposed: bool = False,
):
    assert matrix in ["A", "B"], "matrix should be either A or B"
    dtype_bits = DataType(dtype).bits
    if dtype_bits == 32:
        if matrix == "B" and transposed:
            transform_func = ldmatrix_32x4_to_shared_16x8_layout_b
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx, new_col_idx
        elif matrix == "A" and not transposed:
            transform_func = ldmatrix_32x4_to_shared_16x8_layout_a
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx, new_col_idx
        else:
            raise ValueError("ldmatrix only supports B transposed and A non-transposed for int8")
    elif dtype_bits == 16:
        transform_func = ldmatrix_32x8_to_shared_16x16_layout
        transform_func_trans = ldmatrix_trans_32x8_to_shared_16x16_layout
        if transposed:
            new_row_idx, new_col_idx = transform_func_trans(row_idx, col_idx)
            return new_row_idx, new_col_idx
        else:
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            return new_row_idx, new_col_idx
    elif dtype_bits <= 8:
        if matrix == "B" and transposed:
            transform_func = ldmatrix_32x16_to_shared_16x32_layout_b
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            pack_factor = 8 // dtype_bits
            return new_row_idx, new_col_idx * pack_factor
        elif matrix == "A" and not transposed:
            transform_func = ldmatrix_32x16_to_shared_16x32_layout_a
            new_row_idx, new_col_idx = transform_func(row_idx, col_idx)
            pack_factor = 8 // dtype_bits
            return new_row_idx, new_col_idx * pack_factor
        else:
            raise ValueError("ldmatrix only supports B transposed and A non-transposed for int8")
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def shared_16x16_to_mma_32x8_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 8) // 2
    return thread_id, 4 * (j // 8) + (i // 8) * 2 + (j % 2)


def shared_16x32_to_mma_32x16_layout(i, j):
    thread_id = 4 * (i % 8) + (j % 16) // 4
    return thread_id, 8 * (j // 16) + (i // 8) * 4 + j % 4


def shared_32x16_to_mma_32x16_layout(i, j):
    thread_id = (i % 16) // 4 + 4 * (j % 8)
    return thread_id, 8 * (j // 8) + (i // 16) * 4 + i % 4


def mma_store_index_map(thread_id, local_id):
    return mma_store_32x8_to_shared_16x16_layout(thread_id, local_id)


def mma_store_index_map_fp64(thread_id, local_id):
    return mma_store_32x2_to_shared_8x8_layout_fp64(thread_id, local_id)


def mfma_store_index_map(thread_id, local_id):
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)


def get_mma_micro_size(dtype: Literal["float16", "int8"]):
    # TODO(lei): FP8 related precision support.
    # Basic Tensor Core Matrix Multiply operation Unit
    """
    Return the MMA (Tensor Core) micro-tile dimensions for a given data type.

    This function returns the micro tile sizes (x, y, k) used by MMA/Tensor Core operations.
    - x: tile width in the output/result dimension
    - y: tile height in the output/result dimension
    - k: tile depth in the reduction/K dimension

    Accepted dtype strings include "float16", "int8" and some FP8 identifiers ("float8_e4m3", "float8_e5m2"). For FP8 and int8 types the reduction depth (`k`) is 32; for float16 it is 16.

    Returns:
        tuple[int, int, int]: (micro_size_x, micro_size_y, micro_size_k)
    """
    micro_size_x = micro_size_y = 16
    micro_size_k = 16
    if dtype in {"float8_e4m3", "float8_e5m2", "int8"}:
        micro_size_k = 32
    return micro_size_x, micro_size_y, micro_size_k

from tilelang import tvm as tvm
import tilelang.language as T
from tvm import DataType
from tvm import tir
from tvm.ir import Range
from tvm.tir import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tvm.runtime import convert
from typing import Literal, Callable

from tilelang.utils import is_fragment
from tilelang.utils.language import get_buffer_region_from_load

lift = convert


class TensorCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    N_DIM = 16
    WARP_SIZE = 64
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e4m3fn": "e4m3",
        "float8_e4m3fnuz": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e5m2fn": "e5m2",
        "float8_e5m2fnuz": "e5m2fnuz",
    }

    # k_pack represents the number of elements in a vectorized instruction
    # Detail information can be found in the triton documentation
    # https://github.com/triton-lang/triton/blob/433037206d8870f0b82a3cd669097001084a29ed/third_party/amd/lib/TritonAMDGPUTransforms/AccelerateAMDMatmul.cpp#L419
    k_pack = 1
    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first = False

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        b_preshuffle: bool | None = False,
        thread_var: Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(a_dtype)
        self._initialize_abbrev(a_dtype, b_dtype, accum_dtype)
        self._initialize_local_size(self.M_DIM, self.N_DIM, self.k_dim, self.WARP_SIZE)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_micro_size(self.M_DIM, self.N_DIM, self.k_dim)
        self._initialize_k_pack(k_pack)
        self._initialize_is_m_first(is_m_first)
        self._initialize_b_preshuffle(b_preshuffle)

        self.warp_rows = warp_row_tiles // self.micro_size_x
        self.warp_cols = warp_col_tiles // self.micro_size_y
        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            if a_dtype in ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"]:
                self.k_dim = 32
                return
            a_dtype = DataType(a_dtype)

        if a_dtype.bits == 32:
            self.k_dim = 4
        elif a_dtype.bits in {16, 8}:
            self.k_dim = 16
        else:
            raise ValueError(f"Unsupported a_dtype = {a_dtype}")

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=64):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _dtype_abbrv_lookup(self, dtype):
        s = str(dtype)
        if s.startswith("dtype('") and s.endswith("')"):
            s = s[7:-2]
        if s not in self.dtype_abbrv:
            raise KeyError(f"Unsupported dtype for MACA MMA: {dtype!r}")
        return self.dtype_abbrv[s]

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self._dtype_abbrv_lookup(a_dtype)
        self.b_dtype_abbrv = self._dtype_abbrv_lookup(b_dtype)
        self.accum_dtype_abbrv = self._dtype_abbrv_lookup(accum_dtype)

    def _initialize_mma_prefix(self, k_dim=16):
        in_dtype = self.a_dtype
        M_DIM, N_DIM = self.M_DIM, self.N_DIM

        in_dtype_key = str(in_dtype)
        if in_dtype_key.startswith("dtype('") and in_dtype_key.endswith("')"):
            in_dtype_key = in_dtype_key[7:-2]
        in_dtype_map = {
            "bfloat16": "bf16",
            "float16": "f16",
            "float32": "f32",
            "int8": "i8",
            "int32": "i32",
            "float8_e4m3": "f8",
            "float8_e4m3fn": "f8",
            "float8_e4m3fnuz": "f8",
            "float8_e5m2": "bf8",
            "float8_e5m2fn": "bf8",
            "float8_e5m2fnuz": "bf8",
        }
        in_dtype_abbrv = in_dtype_map[in_dtype_key]

        if in_dtype_abbrv == "f8":
            self.mma_suffix = f"{M_DIM}x{N_DIM}x{k_dim}f8"
        elif in_dtype_abbrv == "bf8":
            self.mma_suffix = f"{M_DIM}x{N_DIM}x{k_dim}bf8"
        elif in_dtype_abbrv == "i8":
            self.mma_suffix = f"{M_DIM}x{N_DIM}x{k_dim}i8"
        elif in_dtype_abbrv == "bf16":
            self.mma_suffix = f"{M_DIM}x{N_DIM}x{k_dim}bf16"
        else:
            self.mma_suffix = f"{M_DIM}x{N_DIM}x{k_dim}{in_dtype_abbrv}"

    def _initialize_micro_size(self, m_dim=16, n_dim=16, k_dim=16):
        self.micro_size_x = m_dim
        self.micro_size_y = n_dim
        self.micro_size_k = k_dim

    def _initialize_k_pack(self, k_pack: int | None = None):
        if k_pack is not None:
            self.k_pack = k_pack

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def _initialize_b_preshuffle(self, b_preshuffle: bool | None = False):
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def get_ldmatrix_index_map(self, is_b=False):
        k_dim = self.k_dim * self.k_pack
        transposed = self.a_transposed if not is_b else self.b_transposed
        if k_dim == 4:
            index_map = shared_4x16_to_local_64x1_layout_B if transposed else shared_16x4_to_local_64x1_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x1_to_4x16_layout_B if transposed else thread_id_shared_access_64x1_to_16x4_layout_A
            )
            if is_b:
                index_map = shared_16x4_to_local_64x1_layout_A if transposed else shared_4x16_to_local_64x1_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x1_to_16x4_layout_A if transposed else thread_id_shared_access_64x1_to_4x16_layout_B
                )
        elif k_dim == 16:
            index_map = shared_16x16_to_local_64x4_layout_B if transposed else shared_16x16_to_local_64x4_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x4_to_16x16_layout_B if transposed else thread_id_shared_access_64x4_to_16x16_layout_A
            )

            if is_b:
                index_map = shared_16x16_to_local_64x4_layout_A if transposed else shared_16x16_to_local_64x4_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x4_to_16x16_layout_A if transposed else thread_id_shared_access_64x4_to_16x16_layout_B
                )
        elif k_dim == 32:
            index_map = shared_16x32_to_local_64x8_layout_B if transposed else shared_16x32_to_local_64x8_layout_A
            reverse_index_map = (
                thread_id_shared_access_64x8_to_16x32_layout_B if transposed else thread_id_shared_access_64x8_to_16x32_layout_A
            )

            if is_b:
                index_map = shared_16x32_to_local_64x8_layout_A if transposed else shared_16x32_to_local_64x8_layout_B
                reverse_index_map = (
                    thread_id_shared_access_64x8_to_16x32_layout_A if transposed else thread_id_shared_access_64x8_to_16x32_layout_B
                )
        else:
            raise ValueError(f"k_dim must be 16 currently but got {k_dim}")

        return index_map, reverse_index_map

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        index_map = IndexMap.from_func(mfma_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def extract_thread_binding(self, thread_id, is_m_first=None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def ldmatrix_a(self, A_local_buf, A_shared_buf: Buffer | BufferRegion, ki, rk=0):
        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        # Leading dimensions (e.g. pipeline stage axis) – empty for 2-D buffers
        A_other = [r.min for r in A_region.region[:-2]]

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (rk * chunk + ki * (k_pack * micro_size_k), warp_m * warp_row_tiles + i * micro_size_x)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[tuple(A_other) + (A_base0 + l + row, A_base1 + r + col)]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * (k_pack * micro_size_k))
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[tuple(A_other) + (A_base0 + l + row, A_base1 + r + col)]

        return _warp_ldmatrix_a(A_local_buf, A_shared_buf, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf, B_shared_buf: Buffer | BufferRegion, ki, rk=0):
        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        # Leading dimensions (e.g. pipeline stage axis) – empty for 2-D buffers
        B_other = [r.min for r in B_region.region[:-2]]

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_col_tiles + j * micro_size_y,
                            rk * chunk + ki * (k_pack * micro_size_k),
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[tuple(B_other) + (B_base0 + l + row, B_base1 + r + col)]

            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * chunk + ki * (k_pack * micro_size_k),
                            warp_n * warp_col_tiles + j * micro_size_y,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[tuple(B_other) + (B_base0 + l + row, B_base1 + r + col)]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        k_pack = self.k_pack
        mma_suffix = self.mma_suffix
        a_dtype, b_dtype, out_dtype = self.a_dtype, self.b_dtype, self.accum_dtype
        compute_a_dtype = a_dtype if local_size_a == 1 else f"{a_dtype}x{local_size_a}"
        compute_b_dtype = b_dtype if local_size_b == 1 else f"{b_dtype}x{local_size_b}"
        compute_out_dtype = out_dtype if local_size_out == 1 else f"{out_dtype}x{local_size_out}"

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * k_pack * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * k_pack * local_size_b if b_is_fragment else 0

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for kp, i, j in T.grid(k_pack, warp_rows, warp_cols):
                T.tvm_mfma(
                    mma_suffix,
                    "row",
                    "row",
                    compute_a_dtype,
                    compute_b_dtype,
                    compute_out_dtype,
                    B_local_buf.data,
                    (b_local_stride + (j * k_pack + kp) * local_size_b) // local_size_b,
                    A_local_buf.data,
                    (a_local_stride + (i * k_pack + kp) * local_size_a) // local_size_a,
                    C_local_buf.data,
                    (i * warp_cols * local_size_out + j * local_size_out) // local_size_out,
                    dtype=compute_out_dtype,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out
        thread_binding = self.get_thread_binding()
        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, N_DIM = self.M_DIM, self.N_DIM
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    if C_buf_dims == 2:
                        C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * N_DIM + col] = C_local_buf[
                            i * (warp_cols * local_size_out) + j * local_size_out + local_id
                        ]
                    else:
                        C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                            i * warp_cols * local_size_out + j * local_size_out + local_id
                        ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id in T.vectorized(local_size_out):
                    row, col = T.meta_var(mfma_store_index_map(tx, local_id))
                    C_buf[
                        (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row, (pid_n * BLOCK_N + warp_n * warp_cols + j) * N_DIM + col
                    ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

    def make_mma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A", "B"], "matrix should be either A or B"
        matrix_is_a: bool = matrix == "A"
        matrix_is_b: bool = matrix == "B"
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None

        k_dim = self.k_dim * self.k_pack

        if k_dim == 4:
            transform_func_sr_a = shared_16x4_to_local_64x1_layout_A
            transform_func_sr_b = shared_16x4_to_local_64x1_layout_A
        elif k_dim == 16:
            transform_func_sr_a = shared_16x16_to_local_64x4_layout_A
            transform_func_sr_b = shared_16x16_to_local_64x4_layout_A
        elif k_dim == 32:
            transform_func_sr_a = shared_16x32_to_local_64x8_layout_A
            transform_func_sr_b = shared_16x32_to_local_64x8_layout_A
        else:
            raise ValueError(f"k_dim must be 0 currently but got {k_dim}")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        transform_func: Callable = None
        if matrix_is_a:
            transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        elif matrix_is_b:
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
        else:
            raise ValueError(f"Unsupported matrix {matrix}")

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        if matrix_is_a:
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r * self.k_pack] if is_sr_axis_order else [micro_size_r * self.k_pack, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk

        warp_s = warp_rows if matrix_is_a else warp_cols
        warp_r = chunk // (micro_size_r * self.k_pack)
        block_s = block_row_warps if matrix_is_a else block_col_warps
        replicate = block_col_warps if matrix_is_a else block_row_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")
        else:
            warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")

        return block_fragment

    def make_mma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.

        Parameters
        ----------
        local_buf : tir.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        shape = local_buf.shape
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)
        assert is_fragment(local_buf), "local_buf must be a fragment"
        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        """
        Convert Buffer/BufferRegion/BufferLoad to a BufferRegion.

        - Buffer -> full-region BufferRegion covering entire shape
        - BufferRegion -> returned as-is
        - BufferLoad -> best-effort convert via get_buffer_region_from_load;
        if scalar, fall back to 1-sized ranges at given indices
        """
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tir.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            # Fallback: scalar load -> 1-sized ranges at indices
            mins = [idx for idx in obj.indices]
            ones = [tir.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")


class TensorCorePreshuffleIntrinEmitter(TensorCoreIntrinEmitter):
    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        k_pack: int | None = None,
        is_m_first: bool | None = False,
        a_preshuffle: bool | None = False,
        b_preshuffle: bool | None = False,
        thread_var: Var | None = None,
    ):
        super().__init__(
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            accum_dtype=accum_dtype,
            a_transposed=a_transposed,
            b_transposed=b_transposed,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            num_elems_per_byte=num_elems_per_byte,
            k_pack=k_pack,
            is_m_first=is_m_first,
            thread_var=thread_var,
        )
        self._initialize_preshuffle(a_preshuffle, b_preshuffle)

    def _initialize_preshuffle(self, a_preshuffle: bool, b_preshuffle: bool):
        if a_preshuffle is not None:
            self.a_preshuffle = a_preshuffle
        if b_preshuffle is not None:
            self.b_preshuffle = b_preshuffle

    def ldmatrix_a(self, A_local_buf, A_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        k_pack = self.k_pack
        is_transposed = self.a_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=False)
        is_global = pid_m is not None and pid_n is not None

        # no preshuffle, use the default implementation
        if self.a_preshuffle is False:
            return super().ldmatrix_a(A_local_buf, A_buf, ki, rk)

        def _warp_ldmatrix_a_global(
            A_local_buf,
            A_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]
            else:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_m * self.block_row_warps + warp_m) * warp_rows + i,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_a_shared(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_m * warp_rows + i,
                        )
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]
            else:
                print(self.a_preshuffle)
                for i in T.serial(warp_rows):
                    for local_id in T.vectorized(k_pack * local_size_a):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (warp_m * warp_rows + i, rk * (chunk // micro_size_k) + ki)
                        A_local_buf[i * k_pack * local_size_a + local_id] = A_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_a_global(A_local_buf, A_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_a_shared(A_local_buf, A_buf, ki, thread_binding, rk)
        )

    def ldmatrix_b(self, B_local_buf, B_buf, ki, rk=0, pid_m=None, pid_n=None):
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        k_pack = self.k_pack
        is_transposed = self.b_transposed
        current_frame = T.KernelLaunchFrame.Current()
        thread_binding = current_frame.get_thread_binding()
        _, reverse_index_map = self.get_ldmatrix_index_map(is_b=True)
        is_global = pid_m is not None and pid_n is not None

        if self.b_preshuffle is False:
            return super().ldmatrix_b(B_local_buf, B_buf, ki, rk, pid_m, pid_n)

        @T.macro
        def _warp_ldmatrix_b_global(
            B_local_buf,
            B_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            (pid_n * self.block_col_warps + warp_n) * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_buf[l, r, row, col]

        @T.macro
        def _warp_ldmatrix_b_shared(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            if is_transposed:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            warp_n * warp_cols + j,
                            rk * (chunk // micro_size_k) + ki,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]
            else:
                for j in T.serial(warp_cols):
                    for local_id in T.vectorized(k_pack * local_size_b):
                        row, col = T.meta_var(reverse_index_map(tx, local_id))
                        l, r = (
                            rk * (chunk // micro_size_k) + ki,
                            warp_n * warp_cols + j,
                        )
                        B_local_buf[j * k_pack * local_size_b + local_id] = B_shared_buf[l, r, row, col]

        return (
            _warp_ldmatrix_b_global(B_local_buf, B_buf, ki, thread_binding, rk)
            if is_global
            else _warp_ldmatrix_b_shared(B_local_buf, B_buf, ki, thread_binding, rk)
        )
# ===== end inline race intrinsics =====



_KERNEL_CACHE = {}
_WORKSPACE_CACHE = {}


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def _moe_forward_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
    block_token=128,
    block_n1=128,
    block_k1=64,
    block_n2=128,
    block_k2=64,
    threads_single=256,
    num_stages=1,
):
    scale = 1.44269504  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32

    num_pairs = (num_blocks_m + 1) // 2

    # ---- 手工 MMA 配置（官方 example_gemm_intrinsics.py 风格）----
    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    chunk = 32
    m_threads = 64 * (block_row_warps * block_col_warps)  # 512

    m_block_M = block_row_warps * warp_row_tiles  # 256
    m_block_N = block_col_warps * warp_col_tiles  # 128

    micro_size = 16
    warp_rows = warp_row_tiles // micro_size  # 4
    warp_cols = warp_col_tiles // micro_size  # 4
    local_size_out = (micro_size * micro_size) // 64  # 4
    n_ki = chunk // micro_size  # 2

    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=dtype,
        b_dtype=dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    input_shape = (total_padded_tokens, hidden)
    intermediate_shape = (total_padded_tokens, intermediate)
    output_shape = (total_padded_tokens, hidden)
    gate_shape = (num_experts, intermediate, hidden)
    up_shape = (num_experts, intermediate, hidden)
    down_shape = (num_experts, hidden, intermediate)
    weights_shape = (total_valid_tokens,)

    @T.prim_func
    def kernel(
        stacked_expert_tokens: T.Tensor(input_shape, dtype),
        gate_w: T.Tensor(gate_shape, dtype),
        up_w: T.Tensor(up_shape, dtype),
        down_w: T.Tensor(down_shape, dtype),
        routed_expert_weights: T.Tensor(weights_shape, T.float32),
        group_sizes: T.Tensor((num_experts,), T.int32),
        group_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_padded_offsets: T.Tensor((num_experts + 1,), T.int32),
        group_idx_for_bx: T.Tensor((num_blocks_m,), T.int32),
        ws: T.Tensor(intermediate_shape, dtype),
        out: T.Tensor(output_shape, dtype),
    ):
        # ---- G_M: gate GEMM, 手工 MMA 合并 256x128 ----
        with T.Kernel(num_pairs, T.ceildiv(intermediate, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_mma_swizzle_layout(A_shared),
                    B_shared: make_mma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))

            if active == 1:
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(hidden, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = stacked_expert_tokens[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = gate_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    for ki in T.serial(0, n_ki):
                        mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                        mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                        mma_emitter.mma(A_local, B_local, C_local)

                mma_emitter.stmatrix(C_local, ws, pid_m=block_start // m_block_M, pid_n=by)

        # ---- U_M: up GEMM + 就地 silu, 手工 MMA 合并 ----
        with T.Kernel(num_pairs, T.ceildiv(intermediate, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_mma_swizzle_layout(A_shared),
                    B_shared: make_mma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))

            if active == 1:
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(hidden, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = stacked_expert_tokens[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = up_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    for ki in T.serial(0, n_ki):
                        mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                        mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                        mma_emitter.mma(A_local, B_local, C_local)

                # 自定义 fragment store：ws = silu(ws) * C_local（复刻 stmatrix global 索引映射）
                tx, warp_n, warp_m = mma_emitter.extract_thread_binding(mma_emitter.get_thread_binding())
                for i, j in T.grid(warp_rows, warp_cols):
                    for local_id in T.vectorized(local_size_out):
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        R = block_start + (warp_m * warp_rows + i) * micro_size + row
                        Cc = by * m_block_N + (warp_n * warp_cols + j) * micro_size + col
                        ws[R, Cc] = (
                            ws[R, Cc]
                            * (1.0 / (1.0 + T.exp2(-ws[R, Cc] * scale)))
                            * C_local[i * (warp_cols * local_size_out) + j * local_size_out + local_id]
                        )

        # ---- D_M: down GEMM, 手工 MMA 合并, rwv select ----
        with T.Kernel(num_pairs, T.ceildiv(hidden, m_block_N), threads=m_threads) as (bx, by):
            A_shared = T.alloc_shared((m_block_M, chunk), dtype=dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((m_block_N, chunk), dtype=dtype, scope="shared.dyn")
            A_local = T.alloc_local((warp_rows * 4,), dtype=dtype)
            B_local = T.alloc_local((warp_cols * 4,), dtype=dtype)
            C_local = T.alloc_local((warp_rows * warp_cols * local_size_out,), dtype=accum_dtype)
            rwv = T.alloc_shared((m_block_M,), dtype=T.float32)

            T.annotate_layout(
                {
                    A_shared: make_mma_swizzle_layout(A_shared),
                    B_shared: make_mma_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(4)

            b0 = bx * 2
            block_start = b0 * block_token
            j1 = T.min(b0 + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > b0, 1, 0)
            eq1 = T.if_then_else(group_idx_for_bx[b0] == group_idx_for_bx[j1], 1, 0)
            active = has1 * eq1

            expert_id = group_idx_for_bx[b0]
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(m_block_M, group_size - token_offset))
            rw_last = total_valid_tokens - 1

            if active == 1:
                for i in T.Parallel(m_block_M):
                    rwv[i] = T.if_then_else(
                        i < actual_rows,
                        routed_expert_weights[T.min(raw_start + token_offset + i, rw_last)],
                        0.0,
                    )

                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(intermediate, chunk), num_stages=num_stages):
                    for i, k in T.Parallel(m_block_M, chunk):
                        A_shared[i, k] = ws[block_start + i, ko * chunk + k]
                    for j, k in T.Parallel(m_block_N, chunk):
                        B_shared[j, k] = down_w[
                            expert_id,
                            by * m_block_N + j,
                            ko * chunk + k,
                        ]
                    for ki in T.serial(0, n_ki):
                        mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                        mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                        mma_emitter.mma(A_local, B_local, C_local)

                tx, warp_n, warp_m = mma_emitter.extract_thread_binding(mma_emitter.get_thread_binding())
                for i, j in T.grid(warp_rows, warp_cols):
                    for local_id in T.vectorized(local_size_out):
                        row, col = T.meta_var(mma_store_index_map(tx, local_id))
                        R = block_start + (warp_m * warp_rows + i) * micro_size + row
                        Cc = by * m_block_N + (warp_n * warp_cols + j) * micro_size + col
                        out[R, Cc] = T.if_then_else(
                            R - block_start < actual_rows,
                            C_local[i * (warp_cols * local_size_out) + j * local_size_out + local_id]
                            * rwv[R - block_start],
                            0.0,
                        )

        # ---- G_S: gate GEMM single (T.gemm, v22 已验证) ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + block_token,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        xs,
                    )
                    T.copy(
                        gate_w[
                            expert_id,
                            by * block_n1 : (by + 1) * block_n1,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        wts,
                    )
                    T.gemm(xs, wts, acc, transpose_B=True)

                for i, j in T.Parallel(block_token, block_n1):
                    if i < actual_rows:
                        ws[block_start + i, by * block_n1 + j] = acc[i, j]

        # ---- U_S: up GEMM + 就地 silu single ----
        with T.Kernel(num_blocks_m, T.ceildiv(intermediate, block_n1), threads=threads_single) as (bx, by):
            xs = T.alloc_shared((block_token, block_k1), dtype=dtype)
            wts = T.alloc_shared((block_n1, block_k1), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n1), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(hidden, block_k1), num_stages=num_stages):
                    T.copy(
                        stacked_expert_tokens[
                            block_start : block_start + block_token,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        xs,
                    )
                    T.copy(
                        up_w[
                            expert_id,
                            by * block_n1 : (by + 1) * block_n1,
                            k * block_k1 : (k + 1) * block_k1,
                        ],
                        wts,
                    )
                    T.gemm(xs, wts, acc, transpose_B=True)

                for i, j in T.Parallel(block_token, block_n1):
                    if i < actual_rows:
                        ws[block_start + i, by * block_n1 + j] = (
                            ws[block_start + i, by * block_n1 + j]
                            * (1.0 / (1.0 + T.exp2(-ws[block_start + i, by * block_n1 + j] * scale)))
                            * acc[i, j]
                        )

        # ---- D_S: down GEMM single ----
        with T.Kernel(num_blocks_m, T.ceildiv(hidden, block_n2), threads=threads_single) as (bx, by):
            hs = T.alloc_shared((block_token, block_k2), dtype=dtype)
            ds = T.alloc_shared((block_n2, block_k2), dtype=dtype)
            acc = T.alloc_fragment((block_token, block_n2), dtype=accum_dtype)

            T.use_swizzle(4)

            j1 = T.min(bx + 1, num_blocks_m - 1)
            has1 = T.if_then_else(j1 > bx, 1, 0)
            eqf = T.if_then_else(group_idx_for_bx[bx] == group_idx_for_bx[j1], 1, 0)
            pm = T.max(bx - 1, 0)
            eqb = T.if_then_else(group_idx_for_bx[pm] == group_idx_for_bx[bx], 1, 0)
            half = bx // 2
            is_even = T.if_then_else(half * 2 == bx, 1, 0)
            covered = T.if_then_else(is_even == 1, has1 * eqf, eqb)

            expert_id = group_idx_for_bx[bx]
            block_start = bx * block_token
            group_size = group_sizes[expert_id]
            raw_start = group_offsets[expert_id]
            padded_start = group_padded_offsets[expert_id]
            token_offset = block_start - padded_start
            actual_rows = T.max(0, T.min(block_token, group_size - token_offset))

            if covered == 0:
                T.clear(acc)
                for k in T.Pipelined(T.ceildiv(intermediate, block_k2), num_stages=num_stages):
                    T.copy(
                        ws[
                            block_start : block_start + block_token,
                            k * block_k2 : (k + 1) * block_k2,
                        ],
                        hs,
                    )
                    T.copy(
                        down_w[
                            expert_id,
                            by * block_n2 : (by + 1) * block_n2,
                            k * block_k2 : (k + 1) * block_k2,
                        ],
                        ds,
                    )
                    T.gemm(hs, ds, acc, transpose_B=True)

                for i, j in T.Parallel(block_token, block_n2):
                    if i < actual_rows:
                        out[block_start + i, by * block_n2 + j] = (
                            acc[i, j] * routed_expert_weights[raw_start + token_offset + i]
                        )
                    else:
                        out[block_start + i, by * block_n2 + j] = 0

    return kernel


def _get_kernel(
    hidden,
    intermediate,
    num_experts,
    total_padded_tokens,
    total_valid_tokens,
    num_blocks_m,
):
    key = (
        int(hidden),
        int(intermediate),
        int(num_experts),
        int(total_padded_tokens),
        int(total_valid_tokens),
        int(num_blocks_m),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _moe_forward_kernel(*key)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _get_workspace(stacked_expert_tokens, intermediate):
    key = (
        int(stacked_expert_tokens.device.index or 0),
        int(stacked_expert_tokens.shape[0]),
        int(intermediate),
        str(stacked_expert_tokens.dtype),
    )
    ws = _WORKSPACE_CACHE.get(key)
    if ws is None:
        ws = torch.empty(
            (int(stacked_expert_tokens.shape[0]), int(intermediate)),
            device=stacked_expert_tokens.device,
            dtype=stacked_expert_tokens.dtype,
        )
        _WORKSPACE_CACHE[key] = ws
    return ws


def run_kernel(
    stacked_expert_tokens,
    gate_w,
    up_w,
    down_w,
    routed_expert_weights,
    group_sizes,
    group_offsets,
    group_padded_offsets,
    group_idx_for_bx,
    out,
):
    hidden = int(stacked_expert_tokens.shape[1])
    intermediate = int(gate_w.shape[1])
    num_experts = int(gate_w.shape[0])
    total_padded_tokens = int(stacked_expert_tokens.shape[0])
    total_valid_tokens = int(routed_expert_weights.shape[0])
    num_blocks_m = int(group_idx_for_bx.shape[0])

    ws = _get_workspace(stacked_expert_tokens, intermediate)
    kernel = _get_kernel(
        hidden,
        intermediate,
        num_experts,
        total_padded_tokens,
        total_valid_tokens,
        num_blocks_m,
    )
    kernel(
        stacked_expert_tokens,
        gate_w,
        up_w,
        down_w,
        routed_expert_weights,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx_for_bx,
        ws,
        out,
    )
