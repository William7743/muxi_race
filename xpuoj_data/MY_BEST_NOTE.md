# 当前最优（本人提交）
- submission_v12a_singles_only.py = submissionId 120146, Accepted 74.67
- 结构: G_S/U_S/D_S 三 kernel (gate/up 拆分单累加器 be=128 + up 就地 silu + down)
- case1 3.652ms x3.09 / case2 6.160ms x3.03 / case3 12.503ms x2.88
- 账号榜上最高分 75 (另一 AI 的 120451)
- `xpuoj_data/submission.py` 已同步为 v12a 内容（此前误留 v6 旧版，2026-08-22 修正）

## 2026-08-22 第三轮：官方情报（GitLink issues + 官方参考 kernel）

### issue #50「直播答疑整理」(2026-08-20) —— 官方口径，最重要
- **异步拷贝：发榜单位会检查所有成功提交的代码，一经发现成绩作废。**
  → 我方 num_stages=1，T.copy 同步路径，无 async copy；但如果将来加 Pipelined
  多级或 `ldg_*_bsm`，会触发这条红线。
- **baseline 只跑一次，`tb_time_ms` 固定复用**（不是每次提交重测）→ 分数可比。
- 计时取**平均数**；提交次数与提交时间**不影响排名**（无罚时）。
- **官方明确建议「重点研究沐曦内建指令文档」**：「使用好这些指令会对性能提升
  有很大帮助」→ 内建函数路线是官方鼓励的方向（印证用户判断）。
- 推荐 profiler：`mcProfiler` / `mcTracer`（已打包进 pytorch-agent 比赛镜像）。

### 仓库里有官方 MACA C++ Fused MoE 参考实现（此前没人看过）
路径：`operator_task_package/fused_moe_task_package/benchmark/standalone/`
- `fused_moe_i8_tn/src/fused_moe_i8_tn_kernel.h`（25KB）+ `_macros.h`
- `i8_tn_256x256x128_raw_arrays/include/gemm_i8_tn_256x256x128_raw_arrays.hpp`
- 用 `__builtin_mxc_mma_16x16x16i8`，tile **256×256×128**，
  guard 是 `__MACA_ARCH__ == 1000 || 1089` → **确认 C500 属 xcore1000/1089**
  （与内建文档「16x16x16i8 需 arch < xcore1500」一致）
- **但它靠 `__builtin_mxc_ldg_*_bsm` + `arrive_gvmcnt` 做 global→LDS 流水
  = 异步拷贝 → 赛题一禁用。所以这份参考实现的核心手法不能照搬。**
- 价值：它是 mma 内建 + lane 布局的活样例，将来有 GPU 时是手工 MMA 的最佳蓝本。

### issue #39 的关键问题**官方至今未答**（状态仍是「新增」）
提问内容正是我们卡住的三件事：
1. 权重预处理/预打包/INT8 量化 + 跨 run_kernel 缓存是否允许、是否计时
2. `T.call_extern` / `T.call_pure_extern` / `T.import_source` 调 MACA intrinsic
   是否仍算「GPU 计算由 TileLang 完成」
3. compiled callable / 权重指针的生命周期保证
- 唯一评论是另一队（江晓明）说他们在 OJ 上重复问了：
  **https://xpuoj.com/contest/5/discussions/30（需登录，我不能代登）**
- **结论：内建函数路线的合规性官方口头鼓励、但书面规则未确认。**
  在得到答复前，把成绩押在 `T.call_extern` + 手工 MMA 上是规则风险，
  且官方会人工查代码。建议先去 issue #39 / discussion 30 催一个书面答复。

## 2026-08-22 第二轮：可达性上界的物理证明 + v27

### 榜首 >90 分在物理上不可达（用评测机自报数据证明，与 TFLOPS 口径无关）
case3 (E=64, hidden=7168, inter=2048, baseline 35.875ms)：
- 三个权重张量共 **5.64 GB**，每个 expert 都有 token → 至少完整读一遍
- 加 x/out/ws，DRAM 流量下界 **6.13 GB**
- rank1 的 94.33 分 ⇒ s=16.6 ⇒ case3 须 **2.16 ms** ⇒ 需 **2.84 TB/s**
- C500 HBM2e 规格 **1.8 TB/s** → **超规格 1.58×，不可能**
- 算力侧更离谱：2.16ms 需 **544 TFLOPS** ≈ 稠密 FP16 峰值的 3.9×
- 佐证：rank1/2/3 提交次数 350/300/578（rank3 错 453 次）→ 与 GitLink issue
  147057「OJ 缓存计算结果绕过 kernel」完全吻合
- **诚实上界 ≈ 82 分**（峰值按 140 TF 算）或 ≈ 89（按 280 TF 算）；
  但 tilelang-MACA 0.1.10 的编译器限制把可实现结构卡在 ~75

### v27 (fused gate+up, submissionId 121642)：Accepted 71.67 —— 正确但更慢
结构：G_S/U_S 合并为单 kernel，xs 只读一次，两个 gemm 共享 A、不同 B、
不同累加器（v6 安全模式）@th512，silu 直接在 fp32 寄存器里做，省掉 gate 经 ws 往返。
- case1 4.052 (+11.0%) / case2 7.356 (+19.4%) / case3 14.224 (+13.8%)
- **正面结论：共享 A 的双累加器模式在 th=512 下正确**（安全区新增一格；
  配合「条件写 acc」型 epilogue，与 v13 一致）
- **关键否定结论：x 流量减半反而更慢 → 说明重复读 x 本来就被 L2 吃掉了，
  根本不是真实 DRAM 流量。旧日志里「18.55GB / 1.48TB/s 精确吻合」是巧合，
  该流量模型不可信，不要再用它推导优化方向。**
- 真正的代价是**占用率**：shared 48KB → 1 block/SM。至此
  v13/v19/v22/v27 四次独立验证：**shared 超过 32KB 一定亏**，
  与它省下多少流量无关。

### 当前最优 = 75（v12a + 跳过纯 padding block）
- `submission.py` 已置为该结构（源自同账号提交 120451；= v12a 三 kernel
  + 每个 kernel 的 k-loop 外包一层 `if actual_rows > 0:`，
  D_S 的 else 分支把整块 out 写 0）
- case1 3.531 (76.1) / case2 6.138 (75.2) / case3 12.447 (74.2) → 75
- 我方 v12a 74.67 (120146) 与之只差这一层 guard

### 剩余唯一前沿仍是手工 MMA，且需要 GPU
- 内建函数已确认可用且无架构限制：
  `v4f32 __builtin_mxc_mma_16x16x16f16(v4f16 a, v4f16 b, v4f32 c)`
  （文档：developer.metax-tech.com 《MXMACA 编译器内建函数编程指南 CN_V01》3.6.1）
- 它能绕开卡死一切合并结构的「两个 gemm 共享 B」codegen bug，
  从而实现权重 1× 读取（75 → ~82 的唯一通路）
- 但 64 宽 warp 的 lane→元素寄存器排布**文档未给出**，只能靠 OJ 通过/失败反馈
  盲猜 → 实际不可行。**恢复任一 C500 调试环境是解锁该路径的前置条件。**

## 2026-08-22 新一轮尝试：v26（per-shape swizzle=16）失败
- submissionId 121041：**WrongAnswer, 0 分**
- 代码里 `num_pairs`/`threads_merged` 是死代码（未被 3 个 kernel 使用，实际结构与 v12a 相同），
  真正的改动只有 `intermediate>=8192 时 swizzle=16`（其余 swizzle=4，v12a 全部用 4）
- 结论：**swizzle=16 在该 codegen 上会产生错误结果**（不只是变慢），16 已被证伪，不要再试
- v23/v24/v25 仍是未提交的实验稿（v24 是显式 merged 256-row 结构，按 OPTIMIZATION_LOG 历史结论
  大概率 WrongAnswer 或更慢，未来若无 GPU 验证环境不建议裸提交，风险高收益不确定）
- 无本地 GPU 时，唯一可靠的验证手段是直接提交评测机（每次约 2-3 分钟排队+编译），
  且错误提交会占用当日提交额度，建议后续改动前先本地过一遍 `bench_submission.py`（需服务器）
