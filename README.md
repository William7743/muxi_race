# Muxi Race - Fused MoE 优化

沐曦「揭榜挂帅」MoE 赛题优化工作区（精简版）。

## 当前最优
- 分数：**74.67**
- 提交：`xpuoj_data/submission.py`（v12a 结构：G_S/U_S/D_S 三 kernel + be=128 + 就地 silu）
- 优化过程：见 `xpuoj_data/OPTIMIZATION_LOG.md`

## 关键结论
- TileLang-MACA 可用子空间：`(128,·) 单累加器 @th256/bk64/be128`
- 手工 MMA / 内建函数路线在评测机不可行（ptx 未注册、模板缺失）
- 常规参数/结构已穷尽，74.67 为当前诚实上限
