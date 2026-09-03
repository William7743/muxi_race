# 功能与正确性测试

正确性验证已内嵌于 `../benchmark/`（依赖关系如下），无需单独脚本：

1. **逐元素参考对照**：`benchmark/bench23.py <候选> <标签> 1,2,3`
   - 自动生成 judge 风格输入（padded token 布局、fp32 routed weights）
   - 以 fp32 torch 参考实现（silu×up → down → routed 加权）为金标准
   - 判定阈值与 XPU-OJ 一致（rtol=atol=0.05），输出 `bad=N`（0 为通过）
2. **压力/竞态测试**：`benchmark/race_stress2.py` + `race_loop*.sh`
   - 随机 expert 尺寸分布（含极端尾块）× 随机数据 seed
   - 每配置 3 次重复，验证 padding/工作区复用的确定性
   - 本版本通过 32/32 seed（case1 20 + case2 6 + case3 6），0 失败 0 NaN

运行方式见 `../README.md` §3。
