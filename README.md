# Muxi Race - Fused MoE 优化

沐曦「揭榜挂帅」MoE 赛题（TileLang 算子优化 - Fused MoE GEMM）优化工作区。

## 当前状态
- **当前最优：75 分**
- 提交代码：`xpuoj_data/submission.py`（submissionId 120451，v21+skip padding 结构）
- 完整优化过程：`xpuoj_data/OPTIMIZATION_LOG.md`
- 目标：**冲榜**（诚实计算极限约 74-75，榜单前列 84+ 多为基线修复前/特殊手段）

## 无 GPU 工作流

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
| 官方评测指南 | https://xpuoj.com/d/2 |

## 关键结论
- TileLang-MACA 可用子空间：`(128,·) 单累加器 @th256/bk64/be128`
- 手工 MMA / 内建函数路线在评测机不可行（ptx 未注册、模板缺失）
- 常规参数/结构已穷尽，75 分为当前诚实上限
