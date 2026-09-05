# v719 local screening, 2026-09-05

Baseline: v718, independently accepted by OJ as submission 139753 / 80.33.
Candidate: v719, changing only E64 Stage2 dispatch to the existing dual-B emitter.
All builder bodies, E16/E32 paths, and the two synchronous launches are unchanged.

Runtime: `/opt/tilelang-metax-v0.1.10` on the shared quarter-C500 instance.
This is not a measurement on the full OJ device. Both routing distributions below
are local fixtures, not verified OJ testcase generators.

Common command arguments to `remote_stage_ab.py`:

```text
--case 3 --stage entry --warmup 1 --iters 1 --rounds 4 --verify-run-kernel
--candidates probe_v718_v716_e64_stage1_terminal_k_only.py probe_v719_v718_e64_stage2_bfrag_only.py
```

- Random run: `--routing alternating64-220 --input-mode random --correctness-repeats 3`.
  Default seed 20260901; three recomputations of the same random input batch,
  not three independent seeds. Before each full-chain and real-entry check,
  workspace/output buffers were poisoned with NaNs. Every result matched v718
  exactly, `max_abs=0`, `bad=0/88080384`.
- Constant run: `--routing synthetic --input-mode constant --correctness-repeats 2`.
  Same poisoning/checking procedure; `max_abs=0`, `bad=0/79822848`.
  This does not establish random-input correctness on the synthetic fixture.

| Fixture | v718 entry median, ms | v719 entry median, ms | Time reduction |
| --- | --- | --- | --- |
| Alternating, random inputs | 8.968960 | 8.915072 | 0.6008% |
| Synthetic, constant inputs | 8.222464 | 8.182912 | 0.4810% |

Each of four forward/reverse-order paired samples favored v719 in each window.
Raw logs retain all samples and statistics. These are small short-test signals,
not a repeatability guarantee or an OJ score claim. The reference is v718's
freshly computed result, not an independent mathematical oracle.

LF-normalized source SHA256:

- v718 tested: `9664d1ab405354b71df30ca14bfe26b73600ecb7ac75932426e44b92c688a4d6`
- v719 tested: `da51c3a541a9d5da55abc0308dcbd35c7bf744b065a3d5ecbd625c6403c34fc7`
- v719 final: `1ac0e5ace6f4059a965420005cdee54db11114fd6d0ccca28f6e68b40e0d7a2f`

Only two header comments changed after GPU testing to record its completion;
the final executable source passed the same complete-source/AST/CPU checks.
The GPU lock was released after both runs. OJ submission remains pending;
the user submits manually in Chrome and supplies an ID for read-only feedback.
