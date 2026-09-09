# NSA112: 合并归约的已完成测试归档

2026-09-09。实际 OJ 最优仍为 NSA109/#141658/86.00；未提交 NSA112。
本轮只是收取此前已启动并完成的 GPU 实验，没有新生成 kernel。

## 设计及资源

NSA112 继承 NSA111 的双块 score 和提前 V0 加载，保持原始16x64 shared
K/V tile与 GEMM。将两个 score 的逐元素 max 合并后做一次 reduce_max，
两个指数概率逐元素相加后做一次 reduce_sum：每对 block 的四次集体归约
减少为两次。FP32 加法结合顺序改变，因此不能只凭数学表达式证明数值通过。

源码 LF SHA256：`6558f5fcd6df83b8ad4d1e8612692f42d13fd104bba74a7bdf883fa476989115`。
audit_nsa112 验证相对111的精确 AST 改动；继承隔离依赖110/111审计。
资源查询111→112：shared3072→2560字节、寄存器80→77、理论驻留21→24块/SM。
查询记录本身未携带源码hash，作为同次运行的辅助证据，不当作实际占用率证明。

## 同批计时

仍是16GB/25%计算资源的C500 sGPU，不是完整64GB实例。第12点、seed137，
同进程109/110/111/112，public/current两种模式，各5轮、前后v159控制。
全部40/40计时通过既定双门限：控制漂移≤3%，相对固定基准[0.92,1.08]。
不是给每个两两比较重复计40个独立样本，整批只有40个。

| 版本 | public中位μs | current中位μs | 两模式控制归一化时间比，109=1 |
|---|---:|---:|---:|
| NSA109 | 62.720 | 63.677 | 1 |
| NSA110 | 64.988 | 65.623 | 1.03295 |
| NSA111 | 64.589 | 65.331 | 1.02785 |
| NSA112 | 61.942 | 62.577 | 0.98527 |

NSA112分别少耗时约1.22%/1.73%；110/111本批较慢，不推荐。
另一个先完成的seed211批次（109/111/112、各3轮）17/18样本合格，
112对109的public/current控制归一化时间比分别0.98777/0.97837。
两个seed方向一致，但尚未进行112全14点的生成代码指纹比较和计时，
不能给出完整分数或替换当前已Accepted版本。

## 正确性与范围

两批计时前共14条完整输出参考断言通过，另有112的16条压力断言通过。
总计30条，其中112为20条，均绑定准确源码hash。
压力覆盖L1024/1040目标路径、L1025/H2回退路径，4种原位更新输入，
包括高幅值、仅首/末槽有效、零Q和交替符号V。它不是所有合法输入的证明。
参考及输入使用PyTorch，提交的GPU运算仍仅为TileLang。

## profiling 注意事项

109/110/111的mcProfiler原始文件及源代码在`pair_profiles/`。
逐kernel的`1_period0_dumped_result.json`记录global writes为16384/29059/90607。
后两者与相同writeback结构预期不符，原因未确定；不得用其推断访存增量或加速。
`report_dumped_result.json`是进程聚合，不是目标kernel单次指令数。
metadata和JSON字段检查PASS不等于计数器隔离/准确性PASS。保留原始计数不删改。

## 下一步及当前暂停点

先保留109为实际成绩基线；112值得下一轮全14点验证，但暂不推荐用户提交。
新提供的kernelgen-flagos技能要求代码优化必须通过KernelGen MCP。
本轮确认项目未配置该服务、工具列表无对应工具。按技能要求暂停新kernel
生成/优化流程，未安装服务、未发送源码到外部服务、未写入凭据。
需要用户配置KernelGen Token后才能继续该技能流程。既有结果收取与归档已完成。

复现CPU核验（仓库根目录）：

```text
python nsa/tests/audit_nsa112.py
python nsa/tests/verify_pair112_results.py
python nsa/tests/summarize_pair_profiles.py
```

计时原始证据：`nsa112.jsonl`、`pairs_quiet_0856.jsonl`及对应log。
派生比较：`initial_112.json`、`quiet_110.json`、`quiet_111.json`、`quiet_112.json`。
`pair112_verification.json`记录源码身份、覆盖和门限计数，不是OJ通过证明。
