# 当前手动 OJ 提交顺序

更新：2026-09-05。已通过已登录 OJ 页面核对源码版本头及成绩；OJ 结果优先于本地切片 GPU 排名。

| 优先级 | 文件 | 提交理由 | 状态 |
| --- | --- | --- | --- |
| 最佳已核实 | [v496](probe_v496_s1_panel3_experts32.py) | [138992](https://xpuoj.com/contest/5/submissions/138992)：Accepted，**79.67**，分项81/79/79。 | 保留为正式基线，不因版本号旧而降级 |
| 下一待测 | [v713](probe_v713_v496_e32_stage1_terminal_k_only.py) | v496仅引入v691的E32 Stage1末K展开；保留E16/E64及全部Stage2，两次launch。两种路由随机精度通过，入口本地改善约1.86%/4.20%。 | 值得用户手动OJ；尚无OJ成绩，不替代v496最佳地位 |
| 已测未升级 | [v691](probe_v691_e32_stage1_split_terminal_k.py) | [139661](https://xpuoj.com/contest/5/submissions/139661)：Accepted，78.33，分项81/78/76。 | 不再推荐重复提交或替代v496 |
| 已测未升级 | [v634](probe_v634_e32_stage2_m64_bfrag_th256.py) | [139669](https://xpuoj.com/contest/5/submissions/139669)：Accepted，78.33，分项81/78/76。 | 不再推荐重复提交或替代v496 |

另已核实[v515 / 139226](https://xpuoj.com/contest/5/submissions/139226)及
[v534 / 139278](https://xpuoj.com/contest/5/submissions/139278)均为79.33、分项80/79/79。
用户最初报告v496为79.33，但138992页面源码及显示分数明确为79.67；记录以查证结果为准。

v691 的新一轮随机验证覆盖拆分链路及真实 `run_kernel(*inputs, out)`，每次运行前将
workspace/out 填为 NaN，连续三轮全部与 v634 输出逐元素一致。有效行必须重新计算，
padding 输出必须清零；缓存只复用已分配空间/编译后的 kernel，不复用历史计算结果。

原始复核记录：
[E32 v691/v701/v702](bench_records/v691_followup/codex_e32_691_701_702_verified.log)。
v713：[alternating随机](bench_records/v691_followup/codex_e32_496_713_entry_random.log)、
[synthetic随机](bench_records/v691_followup/codex_e32_496_713_entry_synthetic.log)、
[synthetic常量复测](bench_records/v691_followup/codex_e32_496_713_entry_synthetic_constant.log)。
第二种路由的4.20%来自常量计时复测；随机窗口有较大长尾，不采用其10.13%中位收益幅度。
完整实验说明见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

v691的本地full指完整预编译kernel链，不含Python入口分派；新加`--stage entry`直接计时
正式`run_kernel`。本地v691仍可能快于v496，但已与OJ方向相反，不能据此升级。

v701/v702/v707 为归因探针，尚无足够稳定的新收益。v711/v712是E16 emitter融合消融，
因v496用例1已为81分，暂不优先提交。其余未在上表列出的新 probe，
均按实验日志标记为待测、无收益或失败，不因为编号较新就推荐提交。

`submission.py` 保持原文件；OJ 由用户手动操作。
