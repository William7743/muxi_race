# v720 E16-only screening, 2026-09-05

Baseline: v719, OJ submission 139764, Accepted / 80.33.
Only executable change: `_get_stage2` expert condition `(32, 64)` becomes
`(16, 32, 64)`, selecting the existing dual-B emitter for E16 as well.
Every Stage1, all builder bodies, E32/E64 paths, and two synchronous launches
remain unchanged. This revisits the existing v711 isolation idea on a new base.

Local runtime: `/opt/tilelang-metax-v0.1.10`, shared quarter-C500 instance.
The routing fixtures are not verified OJ distributions. Measurements time the
real `run_kernel` entry, with one warmup and one invocation per timing sample.

Common `remote_stage_ab.py` arguments:

```text
--case 1 --stage entry --warmup 1 --iters 1 --rounds 4 --verify-run-kernel
```

The first two runs list v719 then v720 as `--candidates`; the third lists
v720 then v719, changing the candidate identities and compilation names as a
diagnostic. Within each run, measurement order alternates forward/reverse.

| Run | Additional arguments | v719 median, ms | v720 median, ms | v720 time reduction | Faster pairs |
| --- | --- | --- | --- | --- | --- |
| Random | `--routing alternating64-220 --input-mode random --correctness-repeats 3` | 2.626176 | 2.612736 | 0.5118% | 2/4 |
| Constant | `--routing synthetic --input-mode constant --correctness-repeats 2` | 2.566528 | 2.545920 | 0.8030% | 2/4 |
| Constant, reversed list | `--routing synthetic --input-mode constant --correctness-repeats 1` | 2.573952 | 2.552832 | 0.8205% | 4/4 |

Important limits:

- Random validation repeats one input batch with default seed 20260901 three
  times; it is not three independent seeds. Constant runs are not additional
  random-input validation. All NaN-poisoned full-chain and real-entry checks
  matched the freshly computed candidate-zero result exactly:
  `max_abs=0`, `bad=0/6291456`. This is equivalence to a tested baseline, not
  a new independent mathematical oracle.
- In the first two windows, v720 was slower on both forward-order pairs and
  faster on both reverse-order pairs. The small median gains must not hide
  that order effect. Reversing the candidate list retained a positive signal,
  but round 3 then had long tails for both versions. Raw logs preserve them.
- The reversed-list log's `speedup_vs_c0=-0.821%` describes v719 relative to
  v720, not v720's speedup. All table percentages use `1 - v720 / v719`.
- These short tests justify an isolated OJ experiment, not a guaranteed score
  increase. v719 remains the same-score, faster-observed baseline; v718 is kept.

LF source SHA256:

- v719 tested: `1ac0e5ace6f4059a965420005cdee54db11114fd6d0ccca28f6e68b40e0d7a2f`
- v720 tested: `a3942810e514278f7dff535b0b2b06c5e2c99b2f12599a85cff7fb4319ce5564`
- v720 final: `2d5605e80220dcecf0e1ae1d86f2edbbb9b60ad2438d2d783f77efe82bb0e774`

Only two header comments changed after testing. Complete-source/AST and CPU
dispatch checks were rerun. CPU mocks included actual E16 H2048/I8192, both
route-weight dtypes and two fresh input calls, with two launches each.
The GPU lock was released; all three raw logs are archived in this directory.
OJ remains pending. The user submits manually in Chrome and supplies the ID.
