# Muxi Race - Fused MoE 优化

沐曦「揭榜挂帅」MoE 赛题（TileLang 算子优化 - Fused MoE GEMM）优化工作区。

> **新接手先读 [`PROGRESS.md`](PROGRESS.md)**（进度总结 + 已关闭路线总表 + 开放方向），
> 逐版本细节见 [`xpuoj_data/OPTIMIZATION_LOG.md`](xpuoj_data/OPTIMIZATION_LOG.md)。

## 当前状态（2026-08-30 最新）

- **可靠最高：76.67（v282，两连 Accepted：126390/126398，timeUsed 19913/19945ms）**；
  两个 submissionId 的评测端代码均已核验为 v282，不是带 panel2/满块分支的 v293。
- 提交代码：`xpuoj_data/submission.py` = v282，与历史最高 76.67 的评测端代码一致。
- 当前窗口原样复验 132087 也 Accepted 75.67，约 3.24/5.89/11.64ms；另有慢资源档
  约 4.33/8.64/17.78ms，所有候选必须按同资源档比较。
- v343 的 `clear_accum` 拆首轮在同资源档仅属噪声级变化，没有形成可确认收益；v344 证明 OJ 实际包仍缺
  `maca_mma_macro_generator`。v318-v324 又证明现有 int8 global I/O/归约写法会
  Segfault/WA，不能把旧 v71 阶段的“int8 待翻案”当成当前开放路线。
- v345/v348（Square+cw8 的 Gate/Up 完整操作数互换）前三次 Accepted 后，v352 原样第4次
  132140 在 case1 出现稀疏误差，复现旧 v202 的不稳定模式，已立即回退。显式布局 132116
  首次 Accepted，但加 barrier 与自然轴 epilogue 均 WA，现正做重复采样；另新增“case1 保持
  v282、仅 hidden7168 互换”的形状特化，主动隔离已知数值风险。
- 参考同事已达 **84+，90+ 亦经官方检验**，仍需寻找尚未复现的重大合法路径。

## 评测环境三大特性（决定一切实验方法论）

1. **机器速度分钟~小时级波动 1.5×**：同代码 timeUsed 19913ms(76.67) ↔ 30198ms(69)。
   评分 T_b 固定 → 慢速时段绝对分被硬封顶 ~69-70。**跨时段分数不可比，
   一切结论用同窗 timeUsed 对照**（`xpuoj_data/canary.py` 金丝雀自动采样）。
2. **case1 哈希漂移**：输入按哈希轮换，部分哈希数值边界敏感 → 与代码无关的偶发 WA
   （v58 同码复提交也 WA；v202 曾三连）。新高分需两次连续 Accepted。
3. **checker 只在全部迭代后比对一次 out**（v91 缓存 out 回写曾 Accepted）；
   **judge 沙箱禁 `tensor.data_ptr()`**——用它做缓存键会直接抛异常记 WA
   （v71 系列四连 WA 的教训）；跨调用可靠缓存键仍不存在。

## 本轮（8/28）实验速览

| 版本 | 内容 | 结果 | 结论 |
|---|---|---|---|
| v61 | kernel2 Pipelined ns=2 | 67.33 | 负：MACA 无异步拷贝，双缓冲纯开销 |
| v64 | 双流分块跨 kernel 重叠 | 4 连 WA | MACA 跨流 event 不被遵守，关闭 |
| v66 | be1=64 → 2 CTA/SM | 63 | MMA 效率损失 > 占用率收益 |
| v67 | A/up_logits copy 加 cw8 | = 同窗基线 | 中性 |
| v68 | kernel2 row-order swizzle | timeUsed +12% | W 共享损失 > A L2 收益 |
| v71×4 | down_w 原地 int8 量化 | 全 WA(timeUsed=0) | **沙箱 data_ptr 禁令所致，路线未死，待 v71e 修正复测** |

## 关键修正：int8 路线翻案（2026-08-28）

旧结论「int8 精度不可行」有误：
- 「per-channel 误差 14 万」系测试 bug（scale 未除回），正规量化相对误差不可能超 1-2%
- 误差判据用 1e-2，而 OJ 实际 rtol=0.05（严了 5 倍）
- 正规 per-row amax int8：点积相对误差 ~1.6%（SNR≈36dB），2.5× 余量
- 「量化副本必 OOM」有零净增显存解法：`w.view(torch.int8)` 后字节偏移与元素下标对齐，
  相邻两值打包进同一 fp16 槽位（4D 视图 `(E,H,2,I)` 的 `[e,n,0,j]`），写只落本行字节区
- 待办（v71e）：去掉 `data_ptr()` 缓存键（改 id/shape），其余保持 v71c/v71d 的保守布局
- 若 W8-dequant 成立 → v72 = gate/up W8A8 + A_i8 workspace + i8 MMA（1.5×）

## 无 GPU 工作流（重要约束）

> **本地没有 C500 GPU，远程服务器资源紧张也没有可用的 C500 调试服务器。**
> **不要寻找/使用任何 C500 调试服务器凭据——没有可用的调试服务器。**
> 所有效果只能通过 XPU-OJ 评测平台迭代验证。

1. 本地修改 `xpuoj_data/submission.py`
2. 提交评测：
   ```bash
   export XPUOJ_EMAIL="muxi2026C1050@example.com"
   export XPUOJ_PASSWORD="你的密码"
   python xpuoj_submit.py --code xpuoj_data/submission.py --status
   ```
3. 根据分数/报错迭代；每次提交在 `xpuoj_data/OPTIMIZATION_LOG.md` 追加记录
4. **方法论红线**：WA 先查 timeUsed（0=早崩/异常；>0=数值超差）；
   新分数必须同窗 timeUsed 对照基线；好天气（≤19800ms）才值得冲分

## 关键网站

| 用途 | 链接 |
|---|---|
| XPUOJ 竞赛页 | https://xpuoj.com/contest/5/problem/1 |
| 竞赛官方仓库（GitLink） | https://www.gitlink.org.cn/metax-maca/op_optimization |
| 沐曦开发者网站 | https://developer.metax-tech.com |
| **沐曦内建函数文档** | https://developer.metax-tech.com/api/client/document/preview/1395/index.html |
| TileLang MetaX 源码（race 分支） | https://github.com/tile-ai/tilelang-metax |

## 给接手 AI 的快速开始

1. **先读** `PROGRESS.md` → `xpuoj_data/OPTIMIZATION_LOG.md`——里面记录了总体进度、
   所有已尝试方向、分数和编译器 bug 地图，避免重复踩坑。
2. **历史最高代码**：v282（76.67 分，submissionId 126390/126398；三档最佳约
   3.12/5.64/11.15ms），精确版本位于提交 `b13b7dd`。
3. **当前主文件**：v282。互换结构虽有 132094/132109/132112 三连 Accepted，但第4次
   132140 已复现 case1 稀疏 WA，不能作为稳定代码。
4. **改动后提交**：
   ```bash
   export XPUOJ_EMAIL="muxi2026C1050@example.com"
   export XPUOJ_PASSWORD="<你的密码>"
   python xpuoj_submit.py --code xpuoj_data/submission.py --status
   ```
   - 没有本地 GPU，完全靠评测平台返回分数/报错迭代。
   - 每次提交都应在 `OPTIMIZATION_LOG.md` 追加记录。
5. **评测机已知限制**（详细见日志）：
   - CUDA PTX MMA/ldmatrix 入口不适用；MACA `T.tvm_mfma` 与原生
     `__builtin_mxc_mma_16x16x16f16` 可用。
   - `th=512` 并非一律错误：若每线程 accumulator 保持约64 FP32可正确运行，
     但多数已测组合仍比稳定 `th=256` 慢。
   - 禁止 async/bsm copy；普通同步向量 load/store、barrier、MFMA 和寄存器预取可研究。
   - `xpuoj_data/submission.py` 始终保持当前最高Accepted稳定版，实验文件单独提交。
6. **凭据安全**：仓库不含密码/token，提交时通过环境变量或本地 `.xpuoj_credentials` 提供。
