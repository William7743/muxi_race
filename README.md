# Muxi Race - Fused MoE 优化

沐曦「揭榜挂帅」MoE 赛题（TileLang 算子优化 - Fused MoE GEMM）优化工作区。

## 当前状态
- **当前最优：75 分**
- 提交代码：`xpuoj_data/submission.py`（submissionId 120451，v21+skip padding 结构）
- 完整优化过程：`xpuoj_data/OPTIMIZATION_LOG.md`
- 目标：**冲榜**（诚实计算极限约 74-75，榜单前列 84+ 多为基线修复前/特殊手段）

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
| 沐曦开发者网站 | https://developer.metax-tech.com |
| **沐曦内建函数文档** | https://developer.metax-tech.com/api/client/document/preview/1395/index.html |
| TileLang MetaX 源码（race 分支） | https://github.com/tile-ai/tilelang-metax |

> **官方提示**：官方提供的 built-in 内建函数文档介绍了沐曦硬件的专用指令。使用好这些指令会对性能提升有很大帮助，建议重点研究这份文档。
> 文档地址：https://developer.metax-tech.com/api/client/document/preview/1395/index.html

## 关键结论
- TileLang-MACA 可用子空间：`(128,·) 单累加器 @th256/bk64/be128`
- 手工 MMA / 内建函数路线在评测机不可行（ptx 未注册、模板缺失）
- 常规参数/结构已穷尽，75 分为当前诚实上限

## 给接手 AI 的快速开始

1. **先读** `xpuoj_data/OPTIMIZATION_LOG.md`——里面记录了所有已尝试方向、分数、编译器 bug 地图，避免重复踩坑。
2. **当前最优**：`xpuoj_data/submission.py`（75 分，submissionId 120451）。
3. **改动后提交**：
   ```bash
   export XPUOJ_EMAIL="muxi2026C1050@example.com"
   export XPUOJ_PASSWORD="<你的密码>"
   python xpuoj_submit.py --code xpuoj_data/submission.py --status
   ```
   - 没有本地 GPU，完全靠评测平台返回分数/报错迭代。
   - 每次提交都应在 `OPTIMIZATION_LOG.md` 追加记录。
4. **评测机已知限制**（详细见日志）：
   - 只有 `T.gemm` 是可用张量核入口；`ptx_mma`/`ptx_ldmatrix` 未注册。
   - `th=256` 是唯一安全线程数；`th=128/512` 会 miscompile。
   - 手工 MMA / 内建函数路线已验证不可行。
   - 常规参数空间已穷尽，当前 75 分接近诚实上限。
5. **凭据安全**：仓库不含密码/token，提交时通过环境变量或本地 `.xpuoj_credentials` 提供。
