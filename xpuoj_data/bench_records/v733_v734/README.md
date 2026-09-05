# v733 / v734: local-loop unroll controls, rejected

Date: 2026-09-05. These are independent probes, not changes to submission.py.
They modify only the E32/H7168/I2048 Stage1 concatenation experiment in
v731/v732. The OJ baseline remains the verified 80.33-point v718/v719/v720;
there is no OJ submission for either new probe.

## Hypothesis and isolated change

The parent concatenation kernels compiled and passed local random checks but
were much slower than v720. Their generated C++ had no explicit unroll hints
on the K16 microloops or lane/local output loops. v733/v734 change exactly five
T.serial calls to T.unroll: steady and terminal ki loops, and the three output
loops. This tests whether these hints rescue that implementation. It does not
assert that dynamic indexing or spilling caused the slowdown.

K64/K32, all buffers, official emitter internals, copies, synchronization,
layouts, pass configuration, arithmetic and dispatch are unchanged from each
parent. Still two current-input kernel launches; no async, BSM, external
computation, global prepacking or result reuse. Other shapes and Stage2 remain
v720. See [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md) for exact source isolation.

## Compilation and correctness

Both compile. The generated C++ gains five pragma-unroll lines and moves the
row-valid condition inside the column loop, an independently checked equivalent
predicate nesting. All addresses and barrier placement are preserved. This is
not proof of final machine-code unrolling, register allocation or occupancy.

Local quarter-C500, E32/H7168/I2048, alternating64-220 routing, raw4544/padded6144,
48 M blocks. Seed20260903, random FP16 input/weights and FP32 route weights.
Three NaN-poisoned recomputations of the same input, both launch_full and actual
run_kernel, are exact relative to the fresh v720 candidate-zero reference:
max_abs=0, bad=0/44040192 for all three candidates. This is not three seeds,
an independent mathematical oracle, or a verified OJ routing distribution.

## Untraced timing

Warmup1/iters1/rounds4, alternating forward/reverse; profiler disabled.

| Candidate | R1 ms | R2 ms | R3 ms | R4 ms | Median ms | Mean ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 | 4.667648 | 4.664320 | 4.693504 | 4.630016 | 4.665984 | 4.663872 |
| v733 K64 | 6.929152 | 6.929664 | 6.877184 | 6.987264 | 6.929408 | 6.930816 |
| v734 K32 | 15.382272 | 15.352576 | 15.359744 | 15.496448 | 15.371008 | 15.397760 |

Paired candidate-minus-v720 deltas, in microseconds:

| Candidate | All paired deltas | Paired median | Wins / losses |
| --- | --- | ---: | --- |
| v733 | +2261.504, +2265.344, +2183.680, +2357.248 | +2263.424 | 0 / 4 |
| v734 | +10714.624, +10688.256, +10666.240, +10866.432 | +10701.440 | 0 / 4 |

Median elapsed-time increases are 48.509% and 229.427%, respectively. Both
lose every round by a large margin. **Do not recommend either for OJ.**
No second fixture or additional profiler run is warranted by this screen.
This does not rule out all possible concatenation implementations; it closes
these exact structures and their five-loop unroll variation. The parents were
measured in a different batch, so do not turn cross-batch differences into a
claim of a small unroll gain or loss.

Raw logs, kept complete:

- `codex_e32_720_733_734_random_entry.log`, SHA256
  `c57bf813d10b8b14e86efc5d38149b785e9241b70f7e6be798a3c87384afd158`.
- `codex_e32_733_734_codegen.log`, SHA256
  `419c7da462bf1f10c3fdc1dc59b2f039b0989fd654f3ba0f52daf1ce76105d37`.

## Source identity and local verification

| Version | Tested SHA256 | Final SHA256 |
| --- | --- | --- |
| v733 | `2e1cd0386fdff81cf6bd29d5f1bf244c4b91200d3623d4a5b09d30e480d7da4b` | `85cf459e6ddcab3913da2ba49766d3794a3037f4120547b1e04ca302ac3fdb71` |
| v734 | `f781e26896180764d193e02d9e437046d855ae8bf2cdac7771547b8a7de1423f` | `025a7426d86fdf463f2b494af2ad4216b8736137463784d6e8b2dd2a05236e4f` |

After testing, only result/header comments and extra trailing blank lines
changed. Restoring those reproduces both tested SHA256 values exactly; complete
ASTs are identical. The CPU audit was rerun on final sources and passed:

`python xpuoj_data/bench_records/v733_v734/audit_v733_v734_cpu.py`

It also reruns the parent source/geometry audit and checks fresh-input forwarding
and two launches for E1/8/16/32/64, FP16/FP32 routes, and non-target E32 shapes.
All owned GPU jobs finished; the codex GPU lock was checked and released.
