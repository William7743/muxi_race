# 给 GPT 的自主探索提示词

你是一名资深的 GPU 算子优化专家，现在接手一个沐曦「揭榜挂帅」MoE 赛题优化任务。请自主探索、持续迭代，目标是突破当前最优分数。

## 背景

- 赛题：TileLang 算子优化 - Fused MoE GEMM（DeepSeek 风格 pre-routed fused expert GEMM）
- 竞赛页：https://xpuoj.com/contest/5/problem/1
- 当前我方最优：**75 分**
- 已知事实：**84+ 分、90+ 分都是官方验证过的真实分数**，说明存在我们尚未掌握的重大合法优化路径
- 同事曾让另一个 GPT 自主跑 4 个多小时达到 84+，但未保留具体方法

## 环境约束（非常重要）

- **本地没有 C500 GPU**
- **远程也没有可用的 C500 调试服务器**
- **只能通过 XPU-OJ 评测平台迭代验证效果**
- 不要寻找/使用 C500 调试服务器凭据——没有可用的调试服务器
- 评测提交后排队约 20-30 分钟，请利用等待时间分析并准备下一版

## 工作区与仓库

- 代码仓库：`git@github.com:William7743/muxi_race.git`
- 当前最优提交：`xpuoj_data/submission.py`（75 分）
- 完整优化历史与踩坑记录：`xpuoj_data/OPTIMIZATION_LOG.md`（**必读**）
- 提交脚本：`xpuoj_submit.py`
- 提交说明：`README_xpuoj.md`
- 官方题目详情：`problem_detail.json`

## 你的任务

1. 先完整阅读 `OPTIMIZATION_LOG.md`，理解已尝试过的方向与失败原因，但**不要被其中的“极限”结论限制**——已知 84+/90+ 真实存在。
2. 持续修改 `xpuoj_data/submission.py`，通过 `xpuoj_submit.py` 提交到评测平台，根据分数/报错迭代。
3. 每次提交都要在 `OPTIMIZATION_LOG.md` 追加记录（版本、思路、submissionId、分数、报错）。
4. 长时间自主探索（建议至少数小时），不要过早停止。

## 重点研究方向（按优先级）

1. **重新审视评测环境/规则是否已变化**：TileLang 版本、异步拷贝、warp specialization、baseline 是否更新。
2. **权重合并 / persistent kernel**：日志中合并类结构曾 miscompile，但 84+ 可能意味着有能正确编译的写法。
3. **内建函数**：沐曦官方 built-in 文档（见下方链接），`T.call_extern` 通道已验证可用（v38），但尚未用于实际优化。
4. **评测端到端优化**：Python 调度、workspace 复用、缓存、减少每次调用开销。
5. **T.gemm 之外的张量核入口**：`T.tvm_mfma` 能编译但曾 WA，可能有布局/索引问题可修复。

## 关键网站

- XPUOJ 竞赛页：https://xpuoj.com/contest/5/problem/1
- 竞赛官方仓库（GitLink）：https://www.gitlink.org.cn/metax-maca/op_optimization
- 沐曦开发者网站：https://developer.metax-tech.com
- **沐曦内建函数文档**：https://developer.metax-tech.com/api/client/document/preview/1395/index.html
- TileLang MetaX 源码（race 分支）：https://github.com/tile-ai/tilelang-metax

## 提交方法

```bash
export XPUOJ_EMAIL="muxi2026C1050@example.com"
export XPUOJ_PASSWORD="<你的密码>"
python xpuoj_submit.py --code xpuoj_data/submission.py --status
```

注意：仓库不包含密码/token，提交时通过环境变量或本地 `.xpuoj_credentials` 提供。

## 输出要求

- 最终把达到的最高分、对应 submissionId、关键优化点写回 `OPTIMIZATION_LOG.md` 和 `README.md`
- 提交所有代码变更到远程仓库 `main` 分支
