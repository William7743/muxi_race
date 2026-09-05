# v720 / v724 / v725 / v726：mcTracer 分阶段实测

采集日期：2026-09-05。工具：mcTracer 3.7.1.5（采集环境报告）。本记录仅整理已经落盘的 trace，不替代无 trace 的随机输入 A/A、A/B 测试或 OJ 反馈。

## 证据与条件

- 基线单独采集：[v720-1076736.json](v720-1076736.json)，SHA256 `4dff04d0f8d331b7e0affa6fce57ede4420eed279b21ccce58d43f8e4a7d82d6`；运行输出为同目录 `codex_profile_v720_20260905.log`。
- 同批候选采集：[candidates-1077079.json](candidates-1077079.json)，SHA256 `111b055bbf9de04579482c2124397c0ea8383ca518c9bbe2ea6b3485a07e3d81`。
- 候选映射固定为 `c0=v720, c1=v724, c2=v725, c3=v726`。v724 是 E32 Stage1 四份 A fragment 复用；v725/v726 分别仅把当前 K 的 Gate/Input 加载经现有 `up_prefetch` 中转，二者没有叠加。
- 本地 C500，E32，H=7168、I=2048；常量诊断输入，routing 为 `alternating64-220`，raw rows=4544、padded rows=6144、M blocks=48。这不是已知 OJ 输入分布，也不是随机精度证明。
- 通过真实 `run_kernel` entry 路径采集；warmup=1、iters=1、rounds=2，轮次顺序为 forward/reverse；全程开启 tracer。这里的 warmup=1 是一次完整 entry 调用，每次 entry 仍启动 Stage1、Stage2 两个 kernel。
- 过滤设备执行事件 `pid=2, ph="X"`，名称匹配 `stage[12]_stage_ab_case_candidate_<id>_kernel`。按每个候选的时间戳排序并配成 Stage1/Stage2；取最后两对，对应 warmup 后的两轮计时。基线/c0 共 6 对，c1/c2/c3 各 5 对，前面的参考、正确性检查和 warmup 不纳入下表。

## 单位与取值方法

这两份文件的 `ts`、`dur` 以及 `args.submit_ts` 是整数纳秒，不能套用其他 Chrome trace 文件的微秒假设。例：基线第 5 对 Stage1 的 `ts=1788582997488879360`，按纳秒转换为 2026-09-05 04:36:37 UTC，与采集日志相符；`dur=2945024` 即 2945.024 µs，而不是 2945 ms。第 6 对 Stage1 的 `ts=1788582997493706240`。

所有差值先用 Python 整数计算，再除以 1000 显示为 µs，避免把约 10^18 的时间戳先转浮点而丢失低位。定义：

- `gap = Stage2.ts - (Stage1.ts + Stage1.dur)`；仅表示两个已记录 kernel 区间之间的间隔，不命名为“GPU 空闲”或“CPU 开销”。
- `span = Stage1.dur + gap + Stage2.dur`；从 Stage1 开始到 Stage2 结束，不是 CUDA/MACA event 测量窗口，也不是 host wall time。
- `S1 share = sum(Stage1.dur) / sum(Stage1.dur + Stage2.dur)`，不包含 gap。

## 最后两轮：精确分阶段时间

下表单位均为 µs；小数三位由整数 ns 精确换算。

| 采集 / 候选 | 轮次 | Stage1 | gap | Stage2 | span |
| --- | --- | ---: | ---: | ---: | ---: |
| 单独基线 v720 | 1 | 2945.024 | 28.672 | 1679.360 | 4653.056 |
| 单独基线 v720 | 2 | 2931.968 | 26.624 | 1666.048 | 4624.640 |
| 同批 c0 / v720 | 1 | 2926.080 | 33.024 | 1676.032 | 4635.136 |
| 同批 c0 / v720 | 2 | 2940.160 | 32.256 | 1680.896 | 4653.312 |
| 同批 c1 / v724 | 1 | 2944.000 | 27.392 | 1679.104 | 4650.496 |
| 同批 c1 / v724 | 2 | 2924.800 | 32.000 | 1677.312 | 4634.112 |
| 同批 c2 / v725 | 1 | 3235.840 | 33.280 | 1672.192 | 4941.312 |
| 同批 c2 / v725 | 2 | 3238.912 | 28.160 | 1670.912 | 4937.984 |
| 同批 c3 / v726 | 1 | 3305.728 | 32.512 | 1669.632 | 5007.872 |
| 同批 c3 / v726 | 2 | 3252.480 | 28.672 | 1672.960 | 4954.112 |

两轮均值与占比：

| 采集 / 候选 | Stage1 µs | Stage2 µs | gap µs | span µs | S1 share（不含 gap） | S1 / span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单独基线 v720 | 2938.496 | 1672.704 | 27.648 | 4638.848 | 63.725% | 63.345% |
| 同批 c0 / v720 | 2933.120 | 1678.464 | 32.640 | 4644.224 | 63.603% | 63.156% |
| 同批 c1 / v724 | 2934.400 | 1678.208 | 29.696 | 4642.304 | 63.617% | 63.210% |
| 同批 c2 / v725 | 3237.376 | 1671.552 | 30.720 | 4939.648 | 65.949% | 65.539% |
| 同批 c3 / v726 | 3279.104 | 1671.296 | 30.592 | 4980.992 | 66.239% | 65.832% |

在这一小批带 trace 的常量诊断中，v724 的 Stage1 接近 v720；v725/v726 的时间增长集中在 Stage1，Stage2 没有对应增长。只能把它作为“加载改写没有在本诊断中变快”的证据，不能据两轮 trace 推导 OJ 分数或稳定的随机输入性能。

## 间隔不是已证实的 Python 提交延迟

单独基线两轮的 Stage2 `submit_ts` 分别只比 Stage1 开始晚 4253 / 3778 ns，仍比 Stage1 结束早 2940771 / 2928190 ns。也就是说，测得 28672 / 26624 ns 的 gap 时，Stage2 早已提交；这两次样本不支持“Python 等 Stage1 结束后才来得及提交 Stage2”这一解释。

检查这两个 gap 区间，没有与之重叠的 `pid=2, ph="X"` 设备事件。它只说明本次 trace 没记录到该区间的执行事件，不能排除未记录的调度、其他租户或 tracer 影响，不能断言整个 GPU 真正空闲，也不能把 gap 当作可直接消除的 CPU 开销。

基线运行输出的 event 时间为 4691.712 / 4660.992 µs，而对应 span 为 4653.056 / 4624.640 µs，测量窗口之差分别是 38.656 / 36.352 µs。此差值也不等于经过归因的 Python 开销；event 窗口与单纯两个 kernel 的首尾区间不同。

候选 c3 的最早一对有 426782464 ns 的大 gap，属于被排除的计时前阶段，不可混入稳态统计；无需假定其具体原因。

## trace 中的原始资源元数据

各候选同一 stage 的所有事件资源元数据一致。以下直接转录 `args.mem` 和命名字段；寄存器数采用 `registers_per_thread`，不是由 Python/生成源码中的数组个数推算。

| kernel | registers_per_thread | dynamic_shared（bytes） | static_shared（bytes） | private_per_thread（原值） | private_total（原值） | mtreg_occupancy(%)（原值） | shared_memeory_occupancy(%)（原值） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 Stage1（两份采集） | 248 | 32768 | 0 | 0 | 0 | 48 | 50 |
| v724 Stage1 | 256 | 32768 | 0 | 0 | 12 | 50 | 50 |
| v725 Stage1 | 240 | 32768 | 0 | 0 | 0 | 46 | 50 |
| v726 Stage1 | 242 | 32768 | 0 | 0 | 0 | 47 | 50 |
| 所有候选 Stage2 | 152 | 32768 | 0 | 0 | 0 | 29 | 50 |

所有 block 为 `(256,1,1)`，Stage1 grid 为 `(48,16,1)`，Stage2 grid 为 `(48,56,1)`；动态 shared 均为 32 KiB。`shared_memeory_occupancy(%)` 保留 JSON 的原始拼写。

这里不把 `mtreg_occupancy(%)` 解释为 SM 的活跃 warp/CTA occupancy，也不把 shared 字段当作测得的 shared-memory 带宽。`private_total=12` 是 v724 的原始报告值；不能据此断言“每线程 spill 12 bytes”、spill 次数或访问带宽，尤其同一记录的 `private_per_thread` 为 0。v725/v726 的报告寄存器数比基线低，但本次 trace 的 Stage1 反而更慢，说明仅用寄存器数预测收益不可靠。

## 工具能回答什么

这次已经获得的是系统级时间线、kernel 执行区间和资源元数据，未获得硬件计数器形式的实际 HBM/shared 带宽、cache 命中率或 stall 分解。官方将 mcTracer 定位为与 Nsight Systems 类似的系统级时序工具，将 mcProfiler 定位为细粒度 kernel 性能分析工具；更深层的 cycle-trace 另有获取渠道。[官方工具区别说明](https://developer.metax-tech.com/forum/t/profilinggong-ju-zi-xun/223/)

后续计数器分析可参考 [mcProfiler 官方使用手册](https://developer.metax-tech.com/api/client/document/file/211/preview/?file_type=pdf)。本次未使用该手册臆造未采集的指标，也没有把 mcTracer 的成功采集等同于 mcProfiler 计数器采集成功。无 trace 的随机输入 A/A + A/B 最终结果由单独实验记录追加，不与本表混算。

## 本地复算（不运行 GPU）

在仓库根目录用 PowerShell 执行以下自包含代码。Python JSON 解析保留整数时间戳；输出最后两对的原始 ns、均值 µs、占比及元数据。

```powershell
@'
import json, re
from pathlib import Path

root = Path('xpuoj_data/bench_records/v725_v726')
for name in ('v720-1076736.json', 'candidates-1077079.json'):
    events = json.loads((root / name).read_text())['traceEvents']
    kernels = sorted((e for e in events if e.get('pid') == 2
        and e.get('ph') == 'X' and re.fullmatch(
            r'stage[12]_stage_ab_case_candidate_\d+_kernel', e.get('name', ''))),
        key=lambda e: e['ts'])
    for cid in sorted({int(e['name'].split('_')[-2]) for e in kernels}):
        group = [e for e in kernels if f'_candidate_{cid}_' in e['name']]
        assert len(group) % 2 == 0 and len(group) >= 4
        pairs = list(zip(group[-4::2], group[-3::2]))
        rows = []
        for a, b in pairs:
            assert a['name'].startswith('stage1_') and b['name'].startswith('stage2_')
            assert all(isinstance(e[k], int) for e in (a, b) for k in ('ts', 'dur'))
            gap = b['ts'] - a['ts'] - a['dur']
            rows.append((a['dur'], b['dur'], gap))
            print(name, cid, 'S1/S2/gap ns:', rows[-1],
                  'S2 submit - S1 end ns:', b['args']['submit_ts']-a['ts']-a['dur'])
        means = [sum(r[i] for r in rows) / 2 / 1000 for i in range(3)]
        print('mean S1/S2/gap/span us:', means + [sum(means)],
              'S1 kernel share %:', 100 * means[0] / sum(means[:2]))
        for stage in (1, 2):
            samples = [e['args'] for e in group if e['name'].startswith(f'stage{stage}_')]
            keys = ('mem', 'mtreg_occupancy(%)', 'shared_memeory_occupancy(%)', 'grid', 'block')
            meta = {json.dumps({k: a[k] for k in keys}, sort_keys=True) for a in samples}
            assert len(meta) == 1
            print('stage', stage, 'metadata:', meta.pop())
'@ | python -
```
