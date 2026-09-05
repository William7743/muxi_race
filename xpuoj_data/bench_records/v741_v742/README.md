# v741/v742：去掉显式同步的验证与负结果归档

2026-09-05。**v741 未优于同窗两份 v720，也未优于同窗 v739，不推荐 OJ。**
v742 与 v740 的生成源码完全相同，已跳过重复 GPU 测试，不能记为 v742 新实测通过。
本批没有新 OJ ID。

## 设计和生成代码证据

- [v741](../../probe_v741_v739_e32_stage2_short_up_auto_barrier.py)：基于 v739，
  只删除新 Stage2 builder 的唯一显式 steady barrier，短程 next-Up 预取仍在 MMA3 前。
- [v742](../../probe_v742_v740_e32_stage2_short_down_auto_barrier.py)：相同消融用于
  v740 的短程 next-Down 路径。

两版只选择 `E32/H7168/I2048` 新 builder；原 builders、Stage1、非目标 shape、
prologue/terminal、复制顺序、数学、raw/padded 语义、passes 与两次 launch 不变。
没有新增 buffer，保持 M128×N128×K64/256 threads/32 KiB shared；独立 probe
不覆盖 `submission.py`。全部按当前输入重算，无 async/BSM/pipeline、外部计算或结果复用。

[完整生成源码审计](CODEGEN_AUDIT.md)和[原始 codegen 日志](codex_e32_741_742_codegen.log)
证明：v741 恰好删去 v739 生成 C++ 中连续两次同步的一次，剩余三处分别保护首轮／
后续轮 shared 读取、end-K shared 覆盖、terminal shared 读取。v742 则被编译器
补成与 v740 **逐字节相同**的源码，仍为五个静态同步 site。
该证明针对实际捕获的 E32/H7168/I2048/**float32 route** 签名；不等于所有 dtype
或编译器版本都自动同步正确，也不是 ISA、硬件 barrier 或寄存器测量。

## 同批 A/A/B/C 随机测试范围

[完整随机入口日志](codex_e32_720_720_739_741_random_entry.log)中的候选依次是：
`c0=v720-A, c1=v720-B, c2=v739, c3=v741`。A/B 两份 v720 是同一源码的两个独立
候选加载，用来观察本窗口内的重复基线差异，不是两个不同优化版本。

本地 C500、TileLang 0.1.10+maca，E32/H7168/I2048；本地 `alternating64-220`
fixture，4544 valid / 6144 padded 行、48 个 M 块，不冒称已核验的 OJ testcase 分布。
随机种子 `DEFAULT_SEED + case_id = 20260903`，**同一批随机输入三轮复算**，
不是三个随机种子。每轮先将 workspace/out 污染为 NaN，完整 `launch_full` 与真实
`run_kernel` 精度比较均通过，日志为 `max_abs=0.000000, bad=0/44040192`。
`bad` 判据是非有限值或 `abs_diff > 0.05 + 0.05 * abs(reference)`；`max_abs`
只打印六位小数，**没有额外检查 bitwise 相等**，不能据此声称逐位完全一致。
参考是同输入 v720-A 输出，不是独立官方 golden；v742 不在这批四候选中。

`warmup=1, iters=1, rounds=4`，顺序正／反／正／反；以下为完整 **entry** 时间，
不叫作 Stage2 单独计时，也不混入 trace 结果。初始化、编译、NaN 污染与精度核验
在计时外。没有追加第二常量 fixture。

## 完整样本与中位

单位 ms；耗时增加按 `candidate_median / v720-A_median - 1`，不是日志速度比指标。

| 候选 | 中位 | 均值 | 标准差 | 中位耗时相对 A |
| --- | ---: | ---: | ---: | ---: |
| v720-A / c0 | 4.676352 | 4.669760 | 0.013843 | — |
| v720-B / c1 | 4.667136 | 4.665472 | 0.007334 | -0.197% |
| v739 / c2 | 4.692352 | 4.695296 | 0.014144 | +0.342% |
| v741 / c3 | 4.714112 | 4.706112 | 0.021671 | +0.807% |

按 round 1–4 排列的完整样本（ms）：

```text
v720-A [4.673792, 4.678912, 4.646144, 4.680192]
v720-B [4.668672, 4.665600, 4.673792, 4.653824]
v739   [4.699904, 4.680192, 4.716288, 4.684800]
v741   [4.706304, 4.670720, 4.721920, 4.725504]
```

## 同轮配对：不把 A/A 波动当优化

差值单位 µs，按表中左版本减右版本，负数表示左版本更快。
前三行基于日志原始 paired 统计；后两行由同轮原始样本直接相减复算。

| 对照 | 四轮差值 | 成对中位 | 成对均值 | 左版本胜／平／负 |
| --- | --- | ---: | ---: | --- |
| v720-B − v720-A | -5.120, -13.312, +27.648, -26.368 | -9.216 | -4.288 | 3/0/1 |
| v739 − v720-A | +26.112, +1.280, +70.144, +4.608 | +15.360 | +25.536 | 0/0/4 |
| v741 − v720-A | +32.512, -8.192, +75.776, +45.312 | +38.912 | +36.352 | 1/0/3 |
| v741 − v720-B | +37.632, +5.120, +48.128, +71.680 | +42.880 | +40.640 | 0/0/4 |
| v741 − v739 | +6.400, -9.472, +5.632, +40.704 | +6.016 | +10.816 | 1/0/3 |

v741 中位耗时分别比 v720-A、v720-B、v739 高 **0.807%、1.007%、0.464%**。
基线 A/A 本身存在数十微秒的单轮差异，因此不声称这些短测建立了普遍稳定的性能排序；
但当前候选没有获得胜过两份 v720 的证据，足以停止该版并拒绝 OJ 推荐。
生成同步少一条不代表性能必然提升，也不能把入口变化全部归因于 Stage2 同步。

## 保留风险与 CPU 核验

生成 float32 partial epilogue 仍在行有效判断前读取未 clamp 的 raw route weight，
最终 expert 的 padding 可能越过 raw 数组末端，空 raw 数组不能靠后续写零分支证明
安全。这是继承风险，未加入 v723 bounds 修复。随机输入精度比较通过不消除这个问题，
也不证明其他 dtype 的 lowering 内存安全。

[本目录 CPU 审计](audit_v741_v742_cpu.py)已在最终注释版重新执行通过：源码/AST
仅允许删除那条显式 barrier 和重命名 builder/reference；原 builders、数学、
terminal、两种 route dtype、目标/非目标分派、新输入与两次 launch 均保持。
两份 probe 的语法检查和 Ruff 通过。CPU 脚本本身不证明自动同步充分；该项另由
已归档实际生成源码审计覆盖。

## 受测／最终源码 SHA 与精确逆向恢复

以下为 **Python probe** SHA256。v741 的受测值对应本批随机测试；v742 的“受测”
仅指 CPU/codegen 检查版本，不冒称做过独立 GPU 数值或计时测试。

| 版本 | 受测 SHA256 | 最终归档 SHA256 |
| --- | --- | --- |
| v741 | `c6977bb3eca6c7ecf982798595f711b8b04e7358bc6b9c4ce3ddcf3d6ebfe29a` | `7a3c583c55094837f0d41a83da9720a00d0f18171d1c01aedfa953202b4cd1ac` |
| v742 | `8919d703102ae6ea228ae63f7b2de277b5911e790c5c9d6d89b25db0031e3c56` | `5ae6d6f170d137b17eeb97a786e9e335ecea194cc88683392c0cbb71a82b939a` |

变更仅将原来一行 `# Experimental: no GPU or OJ result yet.` 替换成三行真实状态
注释；最终注释版已将数值结论限定为精度比较通过、未检查 bitwise 相等。
已在内存中将这三行精确换回原行，分别恢复上表受测 SHA；没有写回旧文件。
逆向恢复前后全 AST 相等，且从 `import torch` 开始的所有字节完全相同。
两份文件原本及修改后均为 LF；不存在执行逻辑或其他换行归一化变化。

可复核的恢复方法（不写文件）：

```python
lines = final_bytes.splitlines(keepends=True)
prefix = b"# Local E32 same-input" if version == 741 else b"# E32 float32-route"
i = next(i for i, line in enumerate(lines) if line.startswith(prefix))
restored = b"".join(lines[:i] + [b"# Experimental: no GPU or OJ result yet.\n"] + lines[i + 3:])
assert hashlib.sha256(restored).hexdigest() == tested_sha256
assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(final_bytes))
```
