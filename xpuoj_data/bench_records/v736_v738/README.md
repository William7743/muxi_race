# v736–v738：Stage2 尾部同步／短程预取，负结果归档

2026-09-05。三版完整随机输出均通过与 v720 的精度比较，但真实入口计时均 **0 胜 4 负**，
不推荐 OJ。本批没有新 OJ ID，不将本地精度通过或编译成功视作性能升级。

## 设计与历史区别

三版独立基于 v720，仅选择 `E32/H7168/I2048` 的新 Stage2 builder；保持 unsplit
M128×N128×K64、256 threads、32 KiB shared、双 B emitter、两次 kernel launch。
Stage1、非目标 shape、原 builder、terminal K、数学与 raw/padded 语义保持不变；
独立 probe 不覆盖 `submission.py`。全部针对当前输入重算，无 async/BSM/pipeline、
外部计算或结果复用。

| Probe | steady 尾部唯一设计变化 |
| --- | --- |
| [v736](../../probe_v736_v720_e32_stage2_early_barrier.py) | `A3 → barrier → MMA3 → Up copy → Down copy`；不增加 buffer |
| [v737](../../probe_v737_v720_e32_stage2_short_up_prefetch.py) | `A3 → barrier → next-Up fragment → MMA3 → fragment 写 Up → Down copy` |
| [v738](../../probe_v738_v720_e32_stage2_short_down_prefetch.py) | `A3 → barrier → next-Down fragment → MMA3 → Up copy → fragment 写 Down` |

旧 v667/v671 分别在 **ki0 前**预取完整下一 K64 的 Up/Down，保存片段跨四次 MMA；
本批仅跨最后一次 MMA，且作用于当前不拆尾块的 M128 路径，因此不是重复旧结构。
v738 保留原 Up→Down shared 写入顺序，没有同时交换两侧复制顺序。

## 测试边界

- [完整随机入口原始日志](codex_e32_720_736_737_738_random_entry.log)。本地 C500，
  TileLang 0.1.10+maca；E32、H7168、I2048，`alternating64-220` **本地路由 fixture**，
  4544 valid / 6144 padded 行、48 个 M 块，不是经核验的官方 testcase 分布。
- 固定随机种子为 `DEFAULT_SEED + case_id = 20260901 + 2 = 20260903`。
  三轮是**同一批随机输入**复算，不是三组随机种子。
- 每轮先将 workspace/out 污染为 NaN，再分别验证完整 `launch_full` 和真实
  `run_kernel`；日志均打印 `max_abs=0.000000, bad=0/44040192`，参考为同输入 v720
  输出，不是独立官方 golden。判据为有限值及 `abs(diff) <= 0.05 + 0.05*abs(ref)`；
  `max_abs` 只打印六位小数，没有额外 bitwise 相等检查，不能据此宣称逐位完全相同。
  本节统计仅含随机测试；[独立常量 trace](PROFILING.md)不混入以下样本。
- `warmup=1, iters=1, rounds=4`，候选顺序正／反／正／反；计时为完整 **entry**，
  不是 Stage2 单独计时。初始化、编译、NaN 污染与精度核验在计时之外；不混用 trace 时间。

## 四轮入口结果

单位 ms；耗时增加为 `candidate_median / v720_median - 1`，不同于日志的速度比指标。

| 版本 | 中位 | 均值 | 标准差 | 中位耗时增加 |
| --- | ---: | ---: | ---: | ---: |
| v720 | 4.647680 | 4.648704 | 0.013645 | — |
| v736 | 5.969152 | 5.974976 | 0.018870 | +28.433% |
| v737 | 4.762752 | 4.759552 | 0.015484 | +2.476% |
| v738 | 5.650816 | 5.646080 | 0.022487 | +21.584% |

按 round 1–4 排列的完整样本（ms）：

```text
v720 [4.658176, 4.665856, 4.633600, 4.637184]
v736 [5.960704, 5.977600, 6.004736, 5.956864]
v737 [4.738048, 4.774656, 4.773888, 4.751616]
v738 [5.641216, 5.670912, 5.660416, 5.611776]
```

成对差值单位 µs，定义 `candidate − 同轮 v720`，正数为更慢：

| 版本 | 四轮成对差值 | 成对中位 | 成对均值 | 胜／平／负 |
| --- | --- | ---: | ---: | --- |
| v736 | +1302.528, +1311.744, +1371.136, +1319.680 | +1315.712 | +1326.272 | 0/0/4 |
| v737 | +79.872, +108.800, +140.288, +114.432 | +111.616 | +110.848 | 0/0/4 |
| v738 | +983.040, +1005.056, +1026.816, +974.592 | +994.048 | +997.376 | 0/0/4 |

## 静态／编译证据与保留风险

[CPU 审计](audit_v736_v738_cpu.py)和 Ruff 通过；详见
[完整生成代码审计](CODEGEN_AUDIT.md)及[生成原始日志](codex_e32_720_736_737_738_codegen.log)。
v736 生成源码确实只移动一条 barrier；v737/v738 则额外保留编译器同步，静态 site
由 v720 的 3 个变成 4/5 个。按正行 K32 的生成 C++ 控制流计，barrier 调用为
63/94/95 次（分别 v736/v737/v738），不是硬件实测或 ISA 次数。
prefetch 为 `uint4` 全局读、`uint2` shared 写，增加的私有片段是 32 个 FP16 值；
不能据此断言物理寄存器压力、spill 或同步是全部性能损失的原因。

生成的 float32 partial epilogue 仍在行有效条件前读取未 clamp 的 raw route weight；
尾部 padding 可能越过 raw 数组末端，空 raw 数组也不能靠后续写零分支证明安全。
这是继承 v720 的风险，未加入 v723 bounds 修复；随机精度比较通过不能消除该风险。

## 受测源码与最终归档源码

以下是 **probe 源码 SHA256**，不是生成 C++ 的 SHA。受测值保留自本轮 CPU 审计；
测试后仅更新头部结果注释，主线程已反向还原验证受测 SHA，AST 不变。

| 版本 | 受测 SHA256 | 最终归档 SHA256 |
| --- | --- | --- |
| v736 | `dd9e3da2282f880dceb2ac6c3807a800202b17ab36b276d38f0e3d6bda838281` | `af47783ecf022ab6b88e3360f0c24566b98a7e8f4c84db6a60ca242d52675186` |
| v737 | `193ece7ff8fc7a33821e59f3dc8640217348e15154647dad085ed9e3014828d9` | `cbf2cb778d147a9ec822ebe779aac89d990c4fa7607c4d72236ea71e6b34a70f` |
| v738 | `01c3f75016b97d33a7296f3fca68538e7256a0f22df62e5b08645426e85cf5a4` | `f2af2e025374c8e8dc3f90bee43cf2114c22c5ff15781606c71735b60418d017` |

v720 受测与归档 SHA：`2d5605e80220dcecf0e1ae1d86f2edbbb9b60ad2438d2d783f77efe82bb0e774`。
后续[晚 barrier 归因实验 v739/v740](../v739_v740/README.md)也未超过同窗 v720。
