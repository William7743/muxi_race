# v741/v742：编译器自动同步的独立生成代码审计

2026-09-05。独立复核确认：**v741 恰为 v739 的生成源码删去连续两条 barrier
中的一条；v742 的生成源码与 v740 逐字节相同。** 前者保留所需 shared 同步，
后者没有产生新的设备源码变化，因此不安排重复 GPU 测试，也不记为新精度通过。

证据为[本批完整 codegen 日志](codex_e32_741_742_codegen.log)，对照
[v739/v740 完整日志](../v739_v740/codex_e32_739_740_codegen.log)及其
[同步审计](../v739_v740/CODEGEN_AUDIT.md)。本文只审查本地已归档生成 C++，
不包含 GPU 性能、随机正确性、ISA 或硬件计数器结果。

## 完整源码相等性证明

四个 `SOURCE_BEGIN/END` 内的完整源码都已重新计算长度及 SHA256，并与各自
日志元数据一致。这里是**生成 C++** 的 SHA，不是 Python probe SHA。

| 版本 | 字符数 | 静态同步 site | SHA256 |
| --- | ---: | ---: | --- |
| v739 | 19482 | 4 | `98ab83dfd5eb2650ba1bbe7c6967fbdecac71b54a833eb7f137fe8f114c41d1f` |
| v741 | 19459 | 3 | `42b285e3810e6b47287898edcda1186320c6b9f4b47c0156749c9867b596ee63` |
| v740 | 19678 | 5 | `611851b08aadd2bb1e9ee7c76ef745358b2f807fbb11b22f0b4fd918ece9f376` |
| v742 | 19678 | 5 | `611851b08aadd2bb1e9ee7c76ef745358b2f807fbb11b22f0b4fd918ece9f376` |

两对比较都**不需要重命名或去除 kernel 符号**：本批和父批的 codegen review
编号恰好一致。实际断言为：

```python
double = "      __syncthreads();\n      __syncthreads();\n"
single = "      __syncthreads();\n"
assert source739.count(double) == 1
assert source739.replace(double, single, 1) == source741
assert source740 == source742
```

因此 v741 所有全局/shared 地址、复制宽度、片段声明、MMA 顺序和操作数、
terminal、epilogue 与 v739 完全一致，唯一删除项为 23 个字符的一行重复同步。
v742 删除 Python 显式同步后，编译器仍生成完全相同的五个同步 site；不能把
Python 中少一条 `T.sync_threads` 当成实际生成同步减少。

## v741 剩余三处同步的覆盖范围

下列行号属于本批完整日志，不是单独导出的 C++ 文件：

| 同步位置 | 保护的数据依赖 |
| --- | --- |
| 60，steady loop 入口 | 首轮保护 prologue 的 Up/Down shared 写入（42、51）到首次 A/B LDS；后续轮保护上一轮 Up/Down 写入到本轮所有 LDS |
| 148，MMA3 后 | 当前 K 的全部 shared 读取及最后寄存器 MMA 均已结束，之后才在 151 写 Up shared、162 写 Down shared，保护 shared 覆盖的 WAR 依赖 |
| 166，terminal 前 | 保护最后一次 steady shared 写入到 terminal A0/B0/B1 及后续 LDS；terminal A0 在 176、B0 在 181，均位于它之后 |

所有控制条件及循环边界只依赖该 CTA 统一的 metadata，不是线程分歧条件。
next-Up 的全局读取仍在当前 MMA3 前，目的地是独立的私有 `next_up[32]`；
它没有提前覆写 shared，也没有覆写当前 A3/B3 的私有片段。

该特化的有效 CTA 有 32 个 K64 tile、31 个 steady 轮次。按生成 C++ 控制流，
v741 为 `31×2+1=63` 次 barrier 调用，v739 为 `31×3+1=94`；空行 CTA 跳过
shared/MMA 工作并经过外层 terminal barrier。此计数不是实测硬件次数，也不能
证明下游编译器先前是否已经合并 v739 的重复同步。空行控制流可达性本身不证明
后面的 raw-route epilogue 内存安全。

## v742 去重与资源解释边界

v742 仍为五处同步（本批日志 347、367、450、462、474）。其 prologue、提前读取
A0 的位置、Up 写入后的额外同步以及 terminal 控制流，全部逐字节继承 v740；
详见父批审计。正行 K32 的生成 C++ 调用数仍是 95。源码重复是本轮不另跑 v742
的依据，不将 v740 的历史计时或精度结果冒称为 v742 的新实测。

v741/v742 均保留 0/16384 两个 shared offset（32 KiB）；`next_up`/`next_down`
仍为 32 个 FP16 值，prefetch 的 `uint4` 全局读及 `uint2` shared 写未改变。
`reported_n_regs` 与 `reported_n_spills` 都为 null，不能解释为零寄存器、零 spill，
也不能从私有数组大小推导实际 occupancy 或性能。

## 签名局限与继承的 raw-route 风险

本证明只针对日志实际捕获的 **E32 / H7168 / I2048 / float32 route weights**
Stage2 签名、当前 TileLang 0.1.10+maca 编译环境。不得推广成所有 shape、route
dtype 或未来编译器都自动提供相同同步；尤其 float16 route 签名未由本日志核验。

生成 partial-row epilogue 仍在有效行判断前读取未 clamp 的 raw route weight：
**v741 在 271 读取、272 才判断；v742 在 574 读取、575 才判断**。
这些语句与父版本完全一致。最终 expert 的 padding 可能读过 raw 数组末端，
空 raw 数组不能依赖稍后的写零分支证明安全。没有引入 v723 bounds 修复；
同步依赖审计通过和随机输出 exact 都不能消除这个独立的继承风险。
