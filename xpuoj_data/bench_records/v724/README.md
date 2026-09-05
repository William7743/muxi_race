# v724：E32 Stage1 跨 Gate/Up 复用 A fragment，结果中性

独立以 v720 为底，仅 E32 Stage1 改为官方 emitter，保留四份当前 K 的 A 微片段，
供 Gate/Up 两组 MMA 共用。目的是减少第二次 shared-memory 输入读取，不是结果缓存。
保持单 weight shared、GIU、Up 寄存器预取、terminal K、所有其他 shape/Stage2。
此设计与旧 v490 的双 B shared、48 KiB 布局有别，本版 shared 仍为 32 KiB。

## 验证

[可复现 CPU 审计](audit_v724_cpu.py)验证所有旧函数全文不变、新 builder 的严格 AST
变更白名单、每个 accumulator 按 K16 0/1/2/3 累加、四份 A 用前定义、Gate/Up 复用、
原始 copies/guards/barriers/epilogue/pass 不变，布局覆盖与双 dtype 的 shape 分派正确。

[生成代码](codex_e32_724_codegen_stage1.log)也已独立人工审核：

- 第22、24–26行是四个独立 `half[16]` 数组。steady 定义59/78/97/116，
  Up 复用149/163/177/191；terminal 定义215/234/253/272，Up 复用307/321/335/349。
- steady 必要同步在56、133–134、139、196；terminal 在211、289–290、296。
  其中连续重复的 Gate 后 barrier 没有在本版擅自删除。
- 第43行稳态 K0..110，terminal 偏移7104=111×64；每 C 的 K16 次序保持。
  第356–407行保留有效行 FP32 SwiGLU、FP16 rn 舍入与完整输出，padding 跳写不变。
- 源码24,039字符，9个静态同步调用点，共享 offset0/16384；这不是物理寄存器、
  动态 barrier 或 occupancy 测量，不据此杜撰性能根因。

## GPU 快速筛选

[原始随机日志](codex_e32_720_724_entry_random.log)：E32 hidden7168/intermediate2048，
`alternating64-220` 本地 fixture，seed20260903（DEFAULT_SEED20260901 + case2）。
同批输入三轮 NaN 污染后 full/entry 复算，两版均 `max_abs=0,bad=0/44040192`。
这是与 v720 的比较，不是独立数学 oracle，不是三个独立随机种子或正式 OJ 数据。

真实入口 warmup1/iters1、四轮正反序：

| 版本 | 中位 ms | 均值 ms | 标准差 ms | 四轮样本 ms |
| --- | --- | --- | --- | --- |
| v720 | 4.685312 | 4.681408 | 0.015533 | 4.681216, 4.698368, 4.656640, 4.689408 |
| v724 | 4.676480 | 4.679232 | 0.021708 | 4.691712, 4.708352, 4.655616, 4.661248 |

中位耗时低约 0.1885%，两对快两对慢，均值只差0.002176 ms。
不足以支持稳定提速，不优先推荐 OJ，不做第二 fixture 扩展。没有 OJ ID。
保留为已经随机通过的结构实验，不能替代80.33分的 v720 或称已突破分数。

## 相关去重与源码身份

连续重复 Gate 后 barrier 的同类删除已经有旧 v707：随机正确，但随机完整链有长尾、
synthetic 常量复测中性（6.132864 vs6.133888 ms）；因此本版没有同时混入同步删减。
v707 为 T.gemm，不能据此担保 emitter+A复用版删减后的同步推导或收益。

GPU测试 probe SHA256：
`ab1df67bb28b0f3a0a82d69a88b8dbff4405438634dbc0f10575f9f07d4c859d`。
测后只更新头部结果注释，执行逻辑不变。没有改 `submission.py`。
最终记录版 SHA256：
`174a3d9aec7db0ed480e1b642daac13093a904c49f419e66d0799020df4aafa4`。
