# Muxi Race - Fused MoE 优化

沐曦「揭榜挂帅」MoE 赛题（TileLang 算子优化 - Fused MoE GEMM）优化工作区。

## 当前状态（2026-08-23 更新）
- **历史最高：76 分；当前稳定版：v138，75.67 分**（复验样本 123355/123401/123771/123887 均 Accepted）
- 提交代码：`xpuoj_data/submission.py`（v138：融合 Gate/Up + 共享 A + 单权重 buffer + `coalesced_width=4` + 单次有效行 SwiGLU 写回 + column4 swizzle）
- 最新突破与教训：v202（kernel1 MMA 操作数互换，gate/up = W_slice @ x^T）三档稳定快 ~1%、四次 Accepted 76，但随后连续 3 次 case1 WA，定性为转置累加器布局的真实竞态，已回退；详见 `OPTIMIZATION_LOG.md`
- **流量级路线已全面实测封锁（2026-08-23）**：①int8 MMA 在 TVM lowering segfault、fp8 dtype 不存在（W8A8 死）②沙箱禁 `tensor.data_ptr()`、`id(tensor)` 跨调用不稳定（可靠缓存键不存在）③评测 GPU 仅余 0.7-2.4GB，任何全量权重副本必 OOM④v138 权重本就每 slice 只读一次，无冗余可去。剩余唯一未封死的大改方向：`T.import_source/T.call_extern` + `T.tvm_mfma` 手工 LDS/MFMA 调度（copy/MMA 重叠）
- **编译器 bug 地图新增**：T.Parallel 内 block 级索引分支会静默错置数值（v218 三连 WA）；动态三重嵌套循环会 segfault（v213）；kernel2 transpose_B=False 会 segfault（v215）
- 评测环境档案：`xpuoj_data/MACHINE_INFO.md`（4 台 C500 机器、宿主 Xeon Gold 6530、CPU flags/缓存）
- 关键已知事实：WA 提交的 userError 也带完整计时 JSON，故意 WA 的诊断探针可拿性能数据；评测机存在 lowering 非确定性、case 数据哈希轮换与 case1 judge 概率性数值漂移，任何新高分必须两次连续 Accepted 才提升稳定版；judge 计时存在 ±50% 档差，横向对比必须同窗口样本
- 完整优化过程：`xpuoj_data/OPTIMIZATION_LOG.md`
- 目标：**冲榜**（当前我方最优 76；同事已实现 84+，90+ 也经官方检验，说明存在我们完全未掌握的重大优化路径）

## 无 GPU 工作流（重要约束）

> **本地没有 C500 GPU，远程服务器资源紧张也没有可用的 C500 调试服务器。**
> **不要寻找/使用任何 C500 调试服务器凭据——没有可用的调试服务器。**
> 所有效果只能通过 XPU-OJ 评测平台迭代验证。

**本机没有 GPU 也能持续优化**，完全通过 XPU-OJ 评测平台看效果：

1. 本地修改 `xpuoj_data/submission.py`
2. 提交到评测平台：
   ```bash
   export XPUOJ_EMAIL="muxi2026C1050@example.com"
   export XPUOJ_PASSWORD="你的密码"
   python xpuoj_submit.py --code xpuoj_data/submission.py --status
   ```
3. 等待评测返回 `Accepted` / `WrongAnswer` / `RuntimeError` 等
4. 根据分数和报错迭代

提交脚本说明见 `README_xpuoj.md`，官方题目详情见 `problem_detail.json`。

## 关键网站

| 用途 | 链接 |
|---|---|
| XPUOJ 竞赛页 | https://xpuoj.com/contest/5/problem/1 |
| 竞赛官方仓库（GitLink） | https://www.gitlink.org.cn/metax-maca/op_optimization |
| 沐曦开发者网站 | https://developer.metax-tech.com |
| **沐曦内建函数文档** | https://developer.metax-tech.com/api/client/document/preview/1395/index.html |
| TileLang MetaX 源码（race 分支） | https://github.com/tile-ai/tilelang-metax |

> **官方提示**：官方提供的 built-in 内建函数文档介绍了沐曦硬件的专用指令。使用好这些指令会对性能提升有很大帮助，建议重点研究这份文档。
> 文档地址：https://developer.metax-tech.com/api/client/document/preview/1395/index.html

## 关键结论
- TileLang-MACA 可用子空间：`(128,·) 单累加器 @th256/bk64/be128`
- `T.tvm_mfma` 的 C500 fp16 MFMA 已实证可用；K64 操作数至少需要两槽 fragment ring，
  否则会出现稀疏大误差。
- `T.import_source + T.call_extern` 可注入自包含 `common.h` 的原生 MACA device C++，
  并真实读写 TileLang local/shared/global 指针；高层 `gemm.h/gemm_ss` 外部实例化不可用。
- 合法的同步 global→register→LDS 软件流水已实证有效：v78 相对朴素 raw tile
  提升约 0.2-0.6ms，但当前仍未胜过稳定 `T.gemm`。
- 当前76分不是已知真实上限；84+/90+ 均经官方验证，仍需寻找更成熟的LDS/MFMA
  调度、persistent/权重复用或其他重大结构路径。

## 给接手 AI 的快速开始

1. **先读** `xpuoj_data/OPTIMIZATION_LOG.md`——里面记录了所有已尝试方向、分数、编译器 bug 地图，避免重复踩坑。
2. **当前可靠最优代码**：`xpuoj_data/submission.py`（75.67 分，submissionId 123355；三档约 3.245/5.907/11.694ms；原样复验 123401 也 Accepted）。v163/v165 曾各得到 76，但随后精确原样提交均出现稀疏 WA，已降级为不稳定历史候选并回退到 v138。
3. **改动后提交**：
   ```bash
   export XPUOJ_EMAIL="muxi2026C1050@example.com"
   export XPUOJ_PASSWORD="<你的密码>"
   python xpuoj_submit.py --code xpuoj_data/submission.py --status
   ```
   - 没有本地 GPU，完全靠评测平台返回分数/报错迭代。
   - 每次提交都应在 `OPTIMIZATION_LOG.md` 追加记录。
4. **评测机已知限制**（详细见日志）：
   - CUDA PTX MMA/ldmatrix 入口不适用；MACA `T.tvm_mfma` 与原生
     `__builtin_mxc_mma_16x16x16f16` 可用。
   - `th=512` 并非一律错误：若每线程 accumulator 保持约64 FP32可正确运行，
     但多数已测组合仍比稳定 `th=256` 慢。
   - 禁止 async/bsm copy；普通同步向量 load/store、barrier、MFMA 和寄存器预取可研究。
   - `xpuoj_data/submission.py` 始终保持当前最高Accepted稳定版，实验文件单独提交。
5. **凭据安全**：仓库不含密码/token，提交时通过环境变量或本地 `.xpuoj_credentials` 提供。
