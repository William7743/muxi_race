# 当前 OJ 验证与候选队列

更新：2026-09-05。已通过已登录 OJ 页面核对源码版本头及成绩；OJ 结果优先于本地切片 GPU 排名。

| 优先级 | 文件 | 提交理由 | 状态 |
| --- | --- | --- | --- |
| 80分保留基线 | [v714](probe_v714_v713_e32_stage2_bfrag_only.py) | [139698](https://xpuoj.com/contest/5/submissions/139698)：Accepted，**80.00**，分项81/80/79，2.594/4.616/9.207 ms。 | 完整提交源码已与仓库归一化逐字核对；保留已取得80分的基线 |
| 同分实验基础 | [v716](probe_v716_v714_e64_stage1_giu_merge_only.py) | [139730](https://xpuoj.com/contest/5/submissions/139730)：Accepted，**80.00**，分项81/80/79，2.596/4.631/9.051 ms。只改E64 Stage1，点3比v714低156微秒、约1.69%。 | 完整源码归一化逐字一致；本次总耗时16.278 vs16.417 ms，作为同分更快的实验底，不宣称跨窗口稳定全面胜出 |
| 历史对照 | [v496](probe_v496_s1_panel3_experts32.py) | [138992](https://xpuoj.com/contest/5/submissions/138992)：Accepted，**79.67**，分项81/79/79。 | 保留作历史对照，当前最高已为v714的80.00 |
| 已测同分 | [v713](probe_v713_v496_e32_stage1_terminal_k_only.py) | [139689](https://xpuoj.com/contest/5/submissions/139689)：Accepted，**79.67**，分项81/79/79，2.582/4.658/9.214 ms。E32耗时比v496低约1.92%，总分未升级。 | 保留为E32隔离优化的实验基础；不宣称超过v496总分 |
| 组件对照 | [v715](probe_v715_v713_e64_stage1_giu_merge_only.py) | v713仅替换E64 Stage1为v527原始GIU＋shared merge。GPU随机三轮正确，入口中位8.984192 vs9.247488 ms，耗时低约2.85%；另一synthetic常量路由低约0.70%。 | 本文件未单独OJ；组件已融合到v716，v716整体独立取得80分 |
| 已填好待用户提交 | [v718](probe_v718_v716_e64_stage1_terminal_k_only.py) | 仅E64 Stage1分派到已有terminal-K builder，不叠加v717。随机三轮精度通过；入口中位8.949376 vs9.017472 ms，耗时低约0.76%；另一synthetic常量窗口低约0.93%。 | 静态/CPU与本地精度已验证；右侧编辑器39327字符/1062行全量回读一致。本轮未点击提交，待用户点击并完成验证；无新OJ ID，不记作已提交 |
| 独立备用 | [v717](probe_v717_v716_e64_stage2_bfrag_only.py) | 仅E64 Stage2分派到已有双B emitter。随机三轮精度通过；入口中位8.955776 vs9.017472 ms，耗时低约0.68%；另一synthetic常量窗口低约0.79%。 | 不与v718叠加，暂作备用；尚未OJ提交，无新ID |
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
v714/v715新增记录：[E32随机](bench_records/v714_v715/codex_e32_713_714_entry_random.log)、
[E64随机](bench_records/v714_v715/codex_e64_713_715_entry_random.log)、
[E64 synthetic常量](bench_records/v714_v715/codex_e64_713_715_entry_synthetic_constant.log)。
两版随机检查均覆盖完整链及真实入口；v715第二种路由只做了常量双轮精度，不记为随机通过。
v716正式结果：[139730精确记录](bench_records/v714_v715/oj_139730_verified.json)，
正式点3个、样例1个单独排除、缺失结果0个，全部pass。
v717/v718：[E64随机](bench_records/v717_v718/codex_e64_716_717_718_entry_random.log)、
[E64 synthetic常量](bench_records/v717_v718/codex_e64_716_717_718_entry_synthetic_constant.log)。
两版均通过NaN污染后的完整链/真实入口检查；另一synthetic路由仅做常量双轮精度，
不能当作该路由随机已过。四轮计时均为预热1次、每轮1次，随机窗口基线有长尾，OJ尚待验证。
完整实验说明见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

v691的本地full指完整预编译kernel链，不含Python入口分派；新加`--stage entry`直接计时
正式`run_kernel`。本地v691仍可能快于v496，但已与OJ方向相反，不能据此升级。

v701/v702/v707 为归因探针，尚无足够稳定的新收益。v711/v712是E16 emitter融合消融，
因v496用例1已为81分，暂不优先提交。其余未在上表列出的新 probe，
均按实验日志标记为待测、无收益或失败，不因为编号较新就推荐提交。

`submission.py` 保持原文件。2026-09-05用户已授权由代理在已登录浏览器中提交候选并读取
OJ反馈，替代此前“仅用户手动提交”的约束；遇到人机验证交由用户完成。每次提交记录版本、
提交ID、分项成绩及精确耗时，先确认精度，再依据OJ反馈决定是否合并；待测候选不预设必快。
