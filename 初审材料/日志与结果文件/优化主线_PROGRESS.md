# MUXI C500 Fused MoE GEMM 优化 — 工作进度总结

> 更新：2026-09-05。本文是面向快速上手的工作总结；逐版本细节见
> `xpuoj_data/OPTIMIZATION_LOG.md`，合规审计见《合规自查说明》。

## 1. 成绩概览

| 项目 | 值 |
|---|---|
| **当前最高分** | **80.33**（三点 81/80/80） |
| 最终提交版本 | **v755**，OJ submissionId **140440**（Accepted，timeUsed 15 917 μs 历史最快） |
| 三 case 用时 | 2.544 / 4.520 / ~8.9 ms（OJ 页面口径） |
| 分数演进 | 76.67 (v293, 8/28) → 78.00 (v404) → 78.67 (v432) → 79.33 (v469, 9/3) → 79.67 (v478, 9/4) → **80.33 (v718/v719/v720/v755, 9/5)** |
| 排名 | 27 位档（78.67 时期）→ 80.33 后名次提升 |
| 评测机 | MetaX C500：64GB HBM、无 cp.async、禁异步拷贝内置、warp=64 |

**80.33 由两份独立代码达成**（v718 = 139753、v755 系 = 140440 前后同分），
排除单次窗口运气；另有 v719（139764）/v720 等同分档提交。

## 2. 最终架构（v755）核心要素

基座 v469→v478，叠加三层（全部纯同步 TileLang 原语，无任何禁用手段）：

1. **A 操作数 shared 化**：MMA 操作数由 fragment 迁移到 shared，纯 GEMM 吞吐
   26.8 → 87 TFLOPS（3.2×，历史最大单点收益）
2. **三层次无异步 load/MMA 重叠**：
   - Stage1：up 权重寄存器预取（记分板语义同步重叠）+ 双侧 vec4 MMA swizzle 布局
     （`make_mma_swizzle_layout(vecSize=4)` + 匹配 coalesced_width 4/8）
   - Stage2：手写 k0 预装 + range 循环 + 覆盖式拷贝调度（源码 T.Pipelined=0）
   - E64 Stage2：direct TensorCoreIntrinEmitter 双B fragment（官方允许的
     tilelang 布局/编译期接口，微基准快 24%，集成兑现 -0.74%）
3. **per-shape 特化**：E32 Stage1 M32 尾块分支（尾块 padding -96 行）+ panel2；
   E64 terminal-K builder；hidden 分派 k_pack/fast_math/ldgstg_predicated
4. 其他：Stage2 满块快路径 epilogue、padding 行显式清零、up_logits scratch
   （题目建议）、JIT 编译缓存

## 3. 评测环境三大特性（方法论基础）

1. **`tb_time_ms` 为固定基线**（官方答疑 issue 50）——选手分数波动完全来自
   自身用时随机器档位波动（实测 ±1.5% 级）；**跨时段比分数无效，必须比
   timeUsed 同窗对照**
2. **异步禁令 + 成绩作废条款**（issue 50：检查所有成功提交代码）——本方案
   最终代码经静态扫描零命中（T.Pipelined=0、无 import_source/call_extern/
   data_ptr/异步/外部库）
3. **per-shape kernel 分派官方确认允许**（issue 答疑：不涉及写死数值即可）

## 4. 技术搜索空间收敛记录（三方独立验证）

合法空间已被三个独立执行体（zcode / GPT5.6 / 历史探针）系统化扫描至收敛：

| 方向 | 结论 | 证据 |
|---|---|---|
| T.Pipelined ns≥2 | -20%+（MACA 无 cp.async） | 三次独立实验 |
| CUDA 流/event 跨kernel重叠 | 运行时不遵守，竞态 | 4 次复现 |
| 官方 emitter 集成（v487-v490） | 微基准 +24% 但 MoE 无提分 | global/LDS 主导 |
| M256 大块 / bt=64 / be=256 | 寄存器/权重重读放大 | 多轮 |
| int8 量化 + 跨调用复用 | 违反禁令 7/8 + 算术净亏损 | 已弃用 |
| 复制宽度 cw 2/4/8/16/auto | cw8（global→fragment）/cw4（vec4 shared 写）最优 | v479-v486 |
| per-shape th2=512（E64） | 与 panel2 不叠加 | v488 复验 |

正收益已全部并入最终版本。详见 `logs/OPTIMIZATION_LOG.md` 负结果清单。

## 5. 协作分工

- **GPT5.6**：Stage1/Stage2 布局消融战役（v471-v486）、terminal-K builder、
  双B emitter、服务器 harness（lab_run.sh + 随机 seed 认证）
- **zcode（本仓库维护线）**：panel2/k_pack 组合、合规审计（OJ 落盘代码逐行
  扫描 + issue 50/79 条款解读）、GPU_LOCK 协调协议、emitter 基准交付、
  初审材料整理
- 双通道：`/root/AGENT_CHAT.log`（留言）+ `/root/GPU_LOCK`（GPU 互斥）

## 6. 复现指引

见 `source/README.md`；优化全过程与全部正/负结果见
`logs/OPTIMIZATION_LOG.md`（45+ 变体）与 `logs/muxi_race_LOG.md`。
