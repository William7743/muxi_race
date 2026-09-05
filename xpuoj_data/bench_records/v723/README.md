# v723：E32 route-load 边界保护，非提分结论

独立以 v720 为底，仅 E32 Stage2 的 route-weight 地址增加上下界 clamp。
当前属于正确性保护候选：本地检查通过，但短测没有性能收益，尚无 OJ ID。
E16/E64、所有正常 Stage1、MMA 次序、路由乘法、FP16 舍入、padding 写零不变。

## 生成代码发现与设计

v720 的 E32 Stage2 在 FP16 和 FP32 两种 route dtype 下，编译器均把尾块
route load 提前到有效行判断之前。源 Python 中的 `if i < actual_rows` 并未阻止
生成代码先读 raw route 数组。最后一个 expert 的 padding 行可能越过数组逻辑上界。
这是生成代码揭示的风险，不是已经定位到某次 OJ WA 或设备异常。

v723 将两处 route 下标替换为
`T.max(0, T.min(raw_start + token_offset + i, total_valid_tokens - 1))`。
对合法元数据的有效行，此变换是恒等；尾行仍由原 else 分支写零。
生成的 FP16/FP32 源码中，每种 dtype 的 8 个静态标量 route-load 表达式均保留 clamp。
四个向量分量分别使用合法地址，不是仅保护一个向量起始地址。

空数组不能使用上界 -1：E32 `valid=0,padded>0` 选择同 8 tensor 签名、无输入读取
的写零 Stage2，跳过无用 Stage1；`padded=0` 在 workspace/JIT/launch 前返回。
正常输入仍然两次 launch；每次针对当前输入重算，无历史结果复用。
本探针只隔离 E32，不宣称已经修复所有其他 shape 的相同潜在风险。

## 本地检查

- [CPU 审计脚本](audit_v723_cpu.py)：所有旧 builder/helper AST 保持；新 builder
  只改变两处 route index；40 组零输出解释执行、32 组 dtype/shape/空输入分派、
  16,268 有效行 FP16 字节一致、6,260 padding 地址合法。
- [GPU 小型边界检查](codex_v723_edge_check.log)：FP16/FP32 各 4 项全部通过。
  包括空输出不走 workspace/JIT/launch、空路由的 129×129 非整 tile 输出、空 block map、
  以及最后一个有效 token 后紧跟 127 行 padding。正常小输入与 v720 逐位相同。
  测试脚本为 `../../remote_v723_edge_check.py`，不属于提交代码或计时路径。
- [大 E32 随机完整链与入口检查](codex_e32_720_723_entry_random.log)：
  `alternating64-220` 本地 fixture，hidden7168/intermediate2048，padded6144/valid4544。
  同一批随机输入三轮 NaN 污染后复算，两版 full/entry 均
  `max_abs=0,bad=0/44040192`。不是三组随机种子，也不是独立数学参考或 OJ 数据。

## 快速计时

预热 1 次、每次计时 1 次、四轮正反序，计时真实入口：

| 版本 | 中位 ms | 均值 ms | 四轮样本 ms |
| --- | --- | --- | --- |
| v720 | 4.649216 | 4.651200 | 4.640256, 4.666112, 4.641280, 4.657152 |
| v723 | 4.669440 | 4.772352 | 4.679936, 4.658944, 4.628736, 5.121792 |

候选中位耗时增加约 0.435%，两对快、两对慢，另有一个明显长尾。
这不支持提速或稳定无开销的结论；尚未做第二路由计时，不优先作为冲分提交。
保留为边界保护备选，正式最高仍为 v718/v719/v720 的 80.33。

## 可复核来源

- [FP16 首轮生成代码](codex_e32_723_codegen_fp16.log)
- [v720/v723 FP16 生成代码](codex_e32_720_723_codegen_fp16.log)
- [v720/v723 FP32 生成代码](codex_e32_720_723_codegen_fp32.log)
- [空路由生成代码](codex_e32_723_codegen_zero.log)：设备函数最终只接收 out，
  没有任何输入读取，静态同步数为 0。
- 生成代码工具：`../../remote_codegen_review.py`（仅诊断，不启动计算 kernel）。
- GPU 测试 probe SHA256：
  `4f749f45ea547ce36a16f143e5667bf7d3c3f27a6efeabf7a3ec1c30f5ac9235`。
- 测后仅把头部“仅静态检查”注释更新为真实 GPU 结果；最终 SHA256：
  `c780850f58c3b29b4663ca8a5aafd5285b05e7d8f8abe4e10b9ce34bf2ca3fe1`。
  执行逻辑与测试版一致。

生成代码中的 builtin 是评测镜像 TileLang 编译产物，本 probe 没有注入外部设备实现。
不使用 async/BSM/pipeline DSL，也不覆盖 `submission.py`。
