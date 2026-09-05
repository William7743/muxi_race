# 当前 OJ 验证与候选队列

更新：2026-09-05。历史条目按各自记录核对；最新v743/v745来自用户截图，上传源码未取回。OJ结果优先于本地切片GPU排名。

**下一次提交[v748](probe_v748_v747_e64_stage1_runtime_m64.py)，仍是待测候选，不替换已知最高基线。**
v745复测截图140316已恢复Accepted80，第二点4530us比相邻v743复测4599us少约1.5%；
原先72.67严重回退未重现，总分未提高。v748继承v745的E32，只扩展E64两阶段runtime
M64/M128与目标空输入保护。本地两路由完整入口检查通过，中位快约3.04%/0.52%，
配对7/8更快；这是OJ试验依据，不保证提高分数。v746/v747保留隔离对照，不同时提交。

| 优先级 | 文件 | 提交理由 | 状态 |
| --- | --- | --- | --- |
| 结构实验中性，不优先提交 | [v724](probe_v724_v720_e32_stage1_a_fragment_reuse.py) | E32 Stage1保留四份A fragment跨Gate/Up复用，shared仍32KiB；同批随机3轮full/entry完全一致。 | 入口中位4.676480 vs4.685312 ms，只差约0.19%，两对快两对慢，不能认定提速；暂无OJ ID |
| 边界保护备选，非提分推荐 | [v723](probe_v723_v720_e32_route_load_bounds.py) | E32 Stage2 route index clamp，FP16/FP32生成源码均保留；空路由写零。大E32随机3轮和8个小型边界测试通过。 | 尚无OJ ID；入口中位4.669440 vs4.649216 ms，慢约0.435%，有单个长尾，不宣称提速；暂不优先提交 |
| 本地负例，不提交 | [v721](probe_v721_v720_e32_stage1_unroll2.py) | E32 Stage1稳态2倍展开，同批随机三轮复算精度一致；入口中位5.802240 vs4.672768 ms，耗时增加约24.17%，四对都慢。 | 关闭、不做第二fixture、不推荐OJ；生成源码21128 vs12945字符、静态sync19 vs9，但不据此推断物理寄存器/动态同步或根因 |
| 本地负例，不提交 | [v722](probe_v722_v720_e32_stage1_unroll3.py) | E32 Stage1稳态3倍展开，同批随机三轮复算精度一致；入口中位6.109696 vs4.672768 ms，耗时增加约30.75%，四对都慢。 | 关闭、不做第二fixture、不推荐OJ；生成源码21321字符/静态sync19，共享offset仍0/16384；不推断具体occupancy或根因 |
| 同分已通过组合底 | [v720](probe_v720_v719_e16_stage2_bfrag_only.py) | [139770](https://xpuoj.com/contest/5/submissions/139770)：Accepted，**80.33**，分项81/80/80，2.549/4.594/8.888 ms，timeUsed16031微秒。只改E16 Stage2，点1比v719少14微秒（约0.546%）。 | 源码仅少末尾一个LF、AST全等；总分未升，总耗时与v719仅差8微秒，保留v718/v719，不称整体稳定更好 |
| 80.33分保留对照 | [v719](probe_v719_v718_e64_stage2_bfrag_only.py) | [139764](https://xpuoj.com/contest/5/submissions/139764)：Accepted，**80.33**，分项81/80/80，2.563/4.616/8.860 ms，timeUsed16039微秒。仅E64 Stage2变化，点3较v718低66微秒（约0.7394%）。 | 源码只比仓库少末尾一个LF，strip相等、AST全等；保留作v720的同分单变量对照 |
| 80.33分保留基线 | [v718](probe_v718_v716_e64_stage1_terminal_k_only.py) | [139753](https://xpuoj.com/contest/5/submissions/139753)：Accepted，**80.33**，分项81/80/80，2.560/4.600/8.926 ms，timeUsed16086微秒。 | 完整源码LF归一化逐字一致；保留与v719同分的正式基线作对照 |
| 80分保留基线 | [v714](probe_v714_v713_e32_stage2_bfrag_only.py) | [139698](https://xpuoj.com/contest/5/submissions/139698)：Accepted，**80.00**，分项81/80/79，2.594/4.616/9.207 ms。 | 完整提交源码已与仓库归一化逐字核对；保留已取得80分的基线 |
| 80分对照 | [v716](probe_v716_v714_e64_stage1_giu_merge_only.py) | [139730](https://xpuoj.com/contest/5/submissions/139730)：Accepted，**80.00**，分项81/80/79，2.596/4.631/9.051 ms。只改E64 Stage1，点3比v714低156微秒、约1.69%。 | 保留作v718单变量对照，当前最高已经是v718的80.33 |
| 历史对照 | [v496](probe_v496_s1_panel3_experts32.py) | [138992](https://xpuoj.com/contest/5/submissions/138992)：Accepted，**79.67**，分项81/79/79。 | 保留作历史对照，不因编号旧而删除 |
| 已测同分 | [v713](probe_v713_v496_e32_stage1_terminal_k_only.py) | [139689](https://xpuoj.com/contest/5/submissions/139689)：Accepted，**79.67**，分项81/79/79，2.582/4.658/9.214 ms。E32耗时比v496低约1.92%，总分未升级。 | 保留为E32隔离优化的实验基础；不宣称超过v496总分 |
| 组件对照 | [v715](probe_v715_v713_e64_stage1_giu_merge_only.py) | v713仅替换E64 Stage1为v527原始GIU＋shared merge。GPU随机三轮正确，入口中位8.984192 vs9.247488 ms，耗时低约2.85%；另一synthetic常量路由低约0.70%。 | 本文件未单独OJ；组件已融合到v716，v716整体独立取得80分 |
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
不能当作该路由随机已过。四轮计时均为预热1次、每轮1次，随机窗口基线有长尾。
v718现已通过OJ：[139753精确记录](bench_records/v717_v718/oj_139753_verified.json)，
三个正式点全部pass，样例1个排除、缺失0个。E16/E32代码没变，跨窗口耗时差不归因于E64改动。
v719：[同批随机三轮复算](bench_records/v719/codex_e64_718_719_entry_random.log)、
[同批synthetic常量两轮复算](bench_records/v719/codex_e64_718_719_entry_synthetic_constant.log)。
后者不能称为随机精度检查；最终发布文件仅更新了头部注释，候选执行逻辑与GPU测试版相同。
v719正式结果：[139764精确记录](bench_records/v719/oj_139764_verified.json)，三个正式点
全部pass，样例1个排除、缺失0个。提交源码与仓库并非LF归一化逐字相等：仅少末尾一个LF；
已核对完整diff与AST。E16/E32路径未改，跨窗口耗时差不归因于E64 Stage2。
v720三轮测试包含同批随机复算、独立synthetic常量fixture、候选加载列表反置诊断；
仅最后一个窗口四对均快，前两个窗口仍为两对快。详细样本及源码注释前后哈希见实验日志。
v720现已独立通过OJ：[139770精确记录](bench_records/v720/oj_139770_verified.json)，
正式三个点全部pass、样例1排除、缺失0；E32/E64路径没变，点2少22微秒/点3多28微秒
不归因于E16改动。v718/v719/v720三份80.33均保留，v720作后续已通过组合底，下一版本尚未选定。
完整实验说明见 [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md)。

v691的本地full指完整预编译kernel链，不含Python入口分派；新加`--stage entry`直接计时
正式`run_kernel`。本地v691仍可能快于v496，但已与OJ方向相反，不能据此升级。

v701/v702/v707 为归因探针，尚无足够稳定的新收益。v711/v712是E16 emitter融合消融，
因v496用例1已为81分，暂不优先提交。其余未在上表列出的新 probe，
均按实验日志标记为待测、无收益或失败，不因为编号较新就推荐提交。

`submission.py` 保持原文件。用户最新流程偏好：**不再操作集成浏览器；代理只提供代码链接，
用户在Chrome手动提交，给出提交ID后由代理只读获取反馈。** 此约定替代前面的代理浏览器
提交流程。每次记录版本、提交ID、分项成绩及精确耗时，先确认精度，再依据OJ反馈决定
是否合并；待测候选不预设必快。

### 2026-09-05：v743 用户截图反馈

- v743 → [140270](https://xpuoj.com/contest/5/submissions/140270)：Accepted，80.33。
  正式点1/2用时2567/4597us，点3只显示9ms，精确值未知；样例2569us单列。
  [截图及精度限定](bench_records/v743/oj_140270_user_report.json)；未取回上传源码校验。
- 第二点相比v720的4594us基本持平，本次未证明提速，不升级为更快OJ基线。
- v745 → [140296](https://xpuoj.com/contest/5/submissions/140296)：用户截图确认
  Accepted，72.67。正式点1=3733us、点2显示7ms、点3显示14ms，后两者精确us未知；
  样例3747us单列。上传源码未核实，[结果记录](bench_records/v745/oj_140296_user_report.json)。
  暂停推广，先手动复提交[v743不可变原代码](https://github.com/William7743/muxi_race/blob/ec774f8d5390f60cf9764b8f03eb01d731631191/xpuoj_data/probe_v743_v723_e32_stage2_runtime_m64.py)
  作当前窗口对照。v744仍不推荐，E64扩展候选仍须独立验证。
- 随后用户发来 [140309](https://xpuoj.com/contest/5/submissions/140309) 截图：Accepted80，
  2568us/4599us/舍入9ms，样例2573us。按刚才复测安排暂关联v743（非明确版本文字/
  源码验证），[记录](bench_records/v743/oj_140309_user_report.json)。前两点回到原范围。
  下一次只复测[v745同一原代码](https://github.com/William7743/muxi_race/blob/b492df528e365561b3e4cd05702d4fac2355e3fc/xpuoj_data/probe_v745_v743_e32_stage1_runtime_m64.py)
  判断回退可重复性，不修改候选后再混作同版结果。
- 随后 [140316](https://xpuoj.com/contest/5/submissions/140316) 截图：Accepted80，正式
  2565us/4530us/舍入9ms，样例2565us，按原样复测安排暂关联v745；未核验上传源码。
  [记录](bench_records/v745/oj_140316_user_report.json)。对相邻v743第二点减少69us约1.5%，
  总分仍80；不声称稳定收益或确定此前回退根因。下一次只测v748，[本地证据](bench_records/v747_v748/README.md)。
