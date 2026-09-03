# XPU-OJ 评测日志与结果文件（赛题一：Fused MoE GEMM）

## 1. 最终提交（v432）

- **提交号：135985　状态：Accepted　得分：78.67　timeUsed：17.535 s**
- 提交时间：2026-09-02；XPU-OJ 账号：muxi2026C1050

### SPJ Report — Testcase #1（experts=16, hidden=2048, intermediate=8192, total_tokens=2272）

```
  Baseline:           11.251000    ms
  User kernel:        2.789000     ms
  Speedup vs base:    4.034       x
  Score ratio:        0.801353      (80.14%)
  Display score:      80            / 100
  Pass:               OK
```

### SPJ Report — Testcase #2（experts=32, hidden=7168, intermediate=2048, total_tokens=4544）

```
  Baseline:           18.610000    ms
  User kernel:        5.065000     ms
  Speedup vs base:    3.674       x
  Score ratio:        0.786061      (78.61%)
  Display score:      78            / 100
  Pass:               OK
```

### SPJ Report — Testcase #3（experts=64, hidden=7168, intermediate=2048, total_tokens=9088）

```
  Baseline:           35.875000    ms
  User kernel:        9.681000     ms
  Speedup vs base:    3.706       x
  Score ratio:        0.787492      (78.75%)
  Display score:      78            / 100
  Pass:               OK
```

评测环境：XPU-OJ 标准环境（PyTorch 2.8.0+metax 3.7.1.5 / TileLang
0.1.10+cuda.gitf549117c / Python 3.12 / C500 GPU）。计时方式：
warmup 10 次 + 100 次迭代取平均，baseline 固定复用。

## 2. 提交历史摘要（关键版本）

| 提交号 | 版本 | 状态 | 得分 | 说明 |
|---|---|---|---|---|
| 126390 | v282 | Accepted | 76.67 | 双 kernel 基础架构（A-shared、融合 SwiGLU、column swizzle）|
| 132574 | v380 | Accepted | 77.00 | + safe-memory legalize 关闭 + vec256 关闭 |
| 133078 | v386 | Accepted | 77.33 | + Stage2 满块无分支 epilogue |
| 134755 | v404 | Accepted | 78.00 | + hidden7168 Stage1 Down 预取、分派重构 |
| 135985 | **v432** | **Accepted** | **78.67** | + enable_fast_math / lower_ldgstg_predicated / **case1 分派全开** |

## 3. 本地 C500 基准数据（复现记录）

环境：C500 GPU、MACA 3.7.x、tilelang-metax v0.1.10、PyTorch 2.8.0+metax、Python 3.12。
计时：CUDA event，warmup 10 + 100 iters 平均；正确性：fp32 参考实现逐元素对照
（rtol=atol=0.05）。

| 版本 | case1 (ms) | case2 (ms) | case3 (ms) | 正确性 |
|---|---|---|---|---|
| v432（最终） | 2.778–2.784 | 5.728–5.816 | 8.993–9.096 | 32/32 随机分布 seed 通过 |
| 历史基线（官方模板改造前） | — | — | — | A-shared 改造前吞吐 26.8 TFLOPS → 改造后 87 TFLOPS |

正确性压力测试（race_stress2 / race_loop）：随机 expert 尺寸分布 × 随机数据
seed，case1 20 seed、case2 6 seed、case3 6 seed，每 seed 3 次重复，
**0 失败、0 NaN**（验证 padding/尾块/工作区复用的安全性）。

## 4. 关键性能数据（Stage 拆分，CUDA event）

| case | Stage1 | Stage2 | 口径 |
|---|---|---|---|
| case1 | 1.900 ms | 1.090 ms | Stage1 ≈108 TFLOPS（计算 roofline ~94%）|
| case2 | 3.744 ms | 2.078 ms | 带宽 ~1.12 TB/s |
| case3 | 5.675 ms | 3.338 ms | Stage2 ≈1.30 TB/s（d2d copy 带宽 1.44 TB/s 的 90%）|

## 5. 完整实验日志

完整优化过程（45+ 变体、十个优化维度的全部实验数据与机制分析）见随包提交的
`实验日志_OPTIMIZATION_LOG.md` 与 GitHub 仓库
`William7743/muxi_race`（PROGRESS.md + xpuoj_data/OPTIMIZATION_LOG.md）。
