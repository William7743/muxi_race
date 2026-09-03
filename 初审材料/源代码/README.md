# Fused MoE GEMM 优化（赛题一 XH-202608）— 源代码说明

## 1. 文件清单

| 文件 | 说明 |
|---|---|
| `submission.py` | **最终提交版本（v432）**，XPU-OJ Accepted 78.67（提交号 135985）。包含 `run_kernel` 评测入口。|
| `benchmark/bench23.py` | 本地三 case 功能+性能基准（自动生成 judge 风格数据、fp32 参考对照、CUDA event 计时）。|
| `benchmark/remote_bench.py` | 基础数据生成与计时库（judge 风格输入布局：padded token、fp32 routed weights、(E+1) offsets）。|
| `benchmark/race_stress2.py` | 正确性压力测试（随机 expert 分布扫描 + 迭代竞态检测）。|
| `benchmark/race_seed1.py` / `race_seed2.py` | 单 seed 崩溃隔离的压力执行器（配合 race_loop*.sh）。|
| `benchmark/speed_bench.py` / `stage_split.py` | 分 Stage 计时与整体测速。|
| `test/` | 功能与正确性测试说明（见 §4）。|

## 2. 环境依赖

- 沐曦 C500 GPU，MACA 3.7.x（与 XPU-OJ 评测环境一致：3.7.1.5）
- TileLang：`tilelang-metax` v0.1.10（judge 构建 `0.1.10+cuda.gitf549117c`）
- PyTorch 2.8.0+metax
- Python 3.12

环境变量（按实际安装路径调整）：

```bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/opt/tilelang-metax-v0.1.10
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
```

## 3. 运行方式

### 3.1 XPU-OJ 提交

将 `submission.py` 全文作为 tilelang.maca-c500 语言提交即可，入口为 `run_kernel`，
函数签名与官方模板完全一致（10 参数，`out` 为唯一 INOUT，padding 行写 0）。

### 3.2 本地功能与性能基准

```bash
cd 源代码/benchmark
# 三 case 全部：功能对照（fp32 参考实现，rtol/atol=0.05）+ 性能（warmup 10 + 100 iters）
python3 bench23.py submission.py v432 1,2,3
```

输出示例（C500 实测）：

```
RESULT v432 case1: 2.784 ms bad=0
RESULT v432 case2: 5.816 ms bad=0
RESULT v432 case3: 9.088 ms bad=0
```

（绝对时间随切片资源配置浮动；XPU-OJ 同代码实测 2.789/5.065/9.681 ms。）

### 3.3 正确性压力测试

```bash
# 随机 expert 分布 × 数据 seed 扫描（验证 padding/尾块/竞态安全性）
./race_loop.sh submission.py cand 20        # case1, 20 seeds
./race_loop2.sh submission.py cand 2 6      # case2, 6 seeds
./race_loop2.sh submission.py cand 3 6      # case3, 6 seeds
```

预期：全部 `bad=0`（本版本已通过 32/32 seed 认证）。

### 3.4 单 Stage 拆分计时（可选，用于 profile）

```bash
python3 stage_split.py submission.py v432   # 输出 stage1/stage2 分周期
```

## 4. 设计要点（详见技术方案文档）

- 双 kernel：Stage1 融合 gate/up GEMM + SwiGLU，Stage2 down GEMM × routed_weight；
- A 操作数走 shared（MACA MFMA 原生供数路径），tile M128×N128×K64、256 线程；
- per-shape JIT 分派 + 编译开关（safe-memory legalize / vectorize256 /
  fast_math / lower_ldgstg_predicated）；
- Stage1 Up 权重同步寄存器预取（无 async/BSM）；
- 尾块 epilogue 显式写 0（IEEE 754 NaN×0 防护）。

## 5. 合规声明

对照官方《赛题一禁用规则说明》八条逐项静态扫描通过：

1. 全部 GPU 计算由 TileLang `T.gemm`/`T.copy` 生成，无外部设备 Kernel；
2. 无任何 PyTorch 数学计算（torch 仅用于 `torch.empty` 工作区分配与 dtype 判断）；
3. 无 Host-Device 异步拷贝；
4. 无跨调用结果缓存/回放（工作区内容每次全量重算，JIT 缓存仅为编译产物）；
5. 无 testcase/生命周期硬编码；
6-8. 未使用 `T.import_source`、`T.call_extern`、mcTlass 或任何外部设备库。

import 清单：`tilelang`、`torch`。完整审计方法见技术方案文档 §7。
