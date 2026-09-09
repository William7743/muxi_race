# NSA 优化工作区

本目录是 `William7743/muxi_race` 仓库中 NSA 的统一入口。**后续 NSA 源码、测试、文档、实验结果都整理在这里**；MOE 代码及 v748 最终交付包保持不变。

## 目录

| 目录 | 内容 |
|---|---|
| `baselines/` | 冻结的历史基线，不直接覆盖 |
| `probes/` | 独立编号的实验候选，不等同于推荐提交版本 |
| `tests/` | 本地精度检查和交替性能比较；PyTorch 仅用于测试/参考/计时 |
| `docs/` | 当前 OJ 接口约定、环境及优化进度 |
| `results/2026-09-07/` | 本轮原始日志、JSONL 结果及无效输入诊断记录 |
| `vendor/public_20260907/` | 官方公开用例与测试脚本快照，附来源说明 |
| `diagnostics/`、`tools/` | 诊断程序和构建辅助材料，不作为 OJ 提交文件 |
| `.local/` | 仅本机 SSH 辅助文件，已忽略，不推送 |

历史路径 `xpuoj_data/nsa_submission.py` 暂时保留兼容副本；后续不再在那里维护 NSA。规范基线副本为 [baselines/nsa_20260817.py](baselines/nsa_20260817.py)。历史文件内的“最优”“109 例通过”等注释仅代表旧记录，不能当作新实例结论。

整理过程统一了文本换行符，基线代码内容未改；文档中的原始下载校验值用于追溯整理前的文件。

## 当前状态

- 低负载复测已完成：NSA114慢约12%，NSA115仅快约0.24%；后者实际设备代码不同，但收益不足以升级。新NSA116/117的V布局与118同步对照均退步。两批共14/14参考检查、54/54合格计时，见[复测](docs/D128_114_115.md)与[布局实验](docs/D128_116_118.md)。本轮无推荐新提交，实际基线仍86.00分。

- 用户已明确同意跳过KernelGen，恢复直接TileLang优化。NSA114（D128直接输出）与NSA115（行倒数）已完成首轮精度对照；8/8参考检查通过，但两批计时分别仅0/12、2/12合格，均无可用配对，不推荐据此提交。详见[本轮实验](docs/D128_114_115.md)。

- 2026-09-09：已核实 **NSA097/#141647 为84分、NSA103/#141648 为85.79分**。新的逐点校准显示大部分分差来自本地代码未变化的路径；同批固定锚点投影对NSA097仍有2分误差。详见 [校准报告及复现](docs/CALIBRATION_097_103.md)，不能将本地估分当作实际成绩。
- 当前已直接核实的 OJ 最高记录为 **[NSA109](probes/probe_nsa109_s2_scalar_probability.py) / #141658 / Accepted / 86.00分**，超过原锚点v236/#14159485.93。相对NSA103的0.21分增加来自本地未变化路径；第10点仍为8μs/88分，不将全部分差算作S2优化收益。详见 [OJ确认及新实验](docs/OJ109_AND_PAIR110.md)。
- NSA112已补齐全14点正确性与代码指纹核验，28/28通过，仅第12点代码改变。该点第三个seed仍快约1.97%，但全批只有41/84计时合格；固定OJ锚点投影未跨整数计分门槛，仍86.00，保留109为正式基线。详见[NSA112结果](docs/PAIR112_RESULTS.md)。KernelGen现已认证，但首次真实优化请求遇到服务端上游401，未生成代码；见[接入及失败记录](docs/KERNELGEN113.md)。**88分目标尚未达到。**
- 64GB实例已不可用，本轮使用用户提供的 C500 **16GB sGPU（25%算力）**。本地性能不等于完整64GB OJ设备性能。最新结果在 `results/2026-09-09/`，最新决策以 [STATUS](docs/STATUS.md) 为准。

以下为2026-09-07接手时的历史记录，不代表最新候选：

- 已归档用户提供的 **OJ #115804（用户报告 82.86 分）**：[原始提交](baselines/oj_115804.py)，[来源与最小差异](baselines/PROVENANCE.md)。其 GPU 计算与原基线一致，仅多一项 int32 输入类型检查。

- [OJ 题面约定](docs/OJ_CONTRACT.md)：`1/sqrt(D)`、causal=1、连续 FP16 输入/输出、连续 int32 indices，`rtol=atol=1e-2`。
- [进度及决策](docs/STATUS.md)：接手时未推荐新的最终提交版本；本任务没有自动提交 OJ。
- 旧基线和 NSA002 都通过了 3 个 smoke 配置 × 2 个 seed，以及 9 个公开 S>1 用例的全输出精度检查。
- NSA002 部分配置更快、大规模 S=8 明显更慢，不能直接整体替换旧基线。
- `invalid_input_diagnostics/` 的崩溃来自不符合题面连续性要求的测试输入，**不能用来否定旧基线正确性**。
- 109 组公开用例不等于已经确认的完整 OJ 测试集合。

## 运行环境和方法

2026-09-07历史实例：完整 C500 64 GB，MACA 3.7.1.5，Python 3.12.11，PyTorch 包报告 `2.8.0+metax3.7.1.3`，TileLang 报告 `0.1.10+cuda.gitf549117c`。该实例现已不可用，最新16GB复测见上述记录。PyTorch 包标签与 MACA 版本是不同字段，未据此判断镜像不匹配或升级系统。

以下命令在 **GPU 服务器上、仓库根目录** 执行。示例路径基于当前 `/opt` 镜像；不需要安装或替换软件。先为当前 shell 设置环境：

```bash
export MACA_PATH=/opt/maca
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:/opt/tilelang-metax-v0.1.10/3rdparty/tvm/python:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/tilelang-metax-v0.1.10/build/lib:/opt/maca/lib:/opt/maca/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=/opt/conda/bin:/opt/maca/bin:/opt/maca/mxgpu_llvm/bin:$PATH

python nsa/tests/smoke_nsa.py nsa/baselines/nsa_20260817.py
python nsa/tests/smoke_nsa.py nsa/probes/probe_nsa002_gather.py
python nsa/tests/compare_multi.py
```

`smoke_nsa.py` 检查三个配置、两个 seed，显式保证五个 tensor 连续，并比较全部输出。其一次预热/一次计时只用于排查，不能直接据此排名。

`compare_multi.py` 默认比较旧基线和 NSA002，使用公开的 9 个 S>1 用例、seed=314、四轮 ABBA 顺序，每次计时批次 10 次调用。结果写入新建的 `nsa/results/local-时间戳/`，不会追加或覆盖已归档结果。可用 `--baseline`、`--candidate`、`--output-dir` 指定路径。事件计时可能包含主机发射间隙，不等同于纯设备 kernel 时间或 OJ 分数。

只做 Python 语法检查（不执行 GPU 测试）：

```bash
python -m compileall -q nsa/baselines nsa/probes nsa/tests nsa/diagnostics nsa/tools
```

## 后续维护

1. 新实验用独立候选文件，注明父版本与唯一改动，不覆盖冻结基线。
2. 修改测试/计时协议时一并记录；区分本地结果、OJ 结果和推测。
3. 每次实验保留正确性结果、每轮原始耗时和环境；通过更多形状、seed 和复测后再升级候选。
4. 提交前检查无密码、token、SSH 私钥或个人连接信息；仅提交经审查的日志。
5. OJ 仅提交候选源码文件，不提交测试脚本或目录；最终候选尚未确定。
