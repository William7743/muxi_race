# v731 / v732: single-GEMM Gate64 + Up64 concatenation

## Hypothesis and exact historical distinction

Both probes start from v720 and specialize only E32/H7168/I2048 Stage1.
Instead of two M128 x N128 accumulators, assemble current Gate64 and Up64 into
one N128 shared tile and accumulate one M128 x N128 matrix. Apply the existing
FP32 SwiGLU to matching halves and store 64 intermediate columns per CTA.
The complete entry still performs two kernels and recomputes current inputs.
No global weight preprocessing, asynchronous copy, external device code, or
historical result reuse is introduced; Stage2 and other shapes stay v720.

v731 uses K64 / 32 KiB shared; v732 uses K32 / 16 KiB shared. Both use 256
threads, official emitter k_pack=1, four row-warps and one column-warp,
warp32 x 128. This changes resource use and CTA count together; lower logical
accumulator size is not a measured occupancy or speedup result. The number of
N-grid CTAs doubles, so input A logical reads also double. Extra A traffic in
the local alternating fixture is 1344 MiB; weight logical reads are unchanged.

This is not an exact repeat of:

- v67: separate Gate/Up N64 GEMMs and separate accumulators, OJ 72.33.
- v98/v99/v102: single N256 concatenation, layout inference failed.
- v355/v357/v367: operand-swapped M256 concatenation, compilation failed.
- v491/v492: single N256 emitter, 4x1 warps; high-level paired epilogue failed,
  manual lane/local epilogue compiled but was much slower (14.367744 versus
  5.499904 ms in its old local comparison).
- v218/v219: globally preconcatenated and cached weights, still two GEMMs;
  historical result caching is not an allowed basis for these probes.

## Attempt 1: high-level paired epilogue

Initial source SHA256:

| Probe | SHA256 |
| --- | --- |
| v731 | `e0d12321f3c73d926647e58b0738689d1a565ef53eba6ed5098cde0fb499e292` |
| v732 | `b81a9c50dd84946ab37a86883c982069eb202a4a487c8dcd371b5c6c782ca52e` |

v731 compilation failed in LayoutInference, `parallel.cc:192`:
`gu_local: (i, j) and (i, j + 64)`. The helper stopped at v731, so **v732 was
not compiled in this attempt**. No GPU numerical/performance result exists
for attempt 1. Raw output is `codex_e32_731_732_codegen_attempt1.log`.

The official emitter mapping does assign j/j+64 to the same thread, with
local index difference 16. Gate local slots are 0..15 and 32..47; Up slots
16..31 and 48..63. This geometric fact is insufficient for the frontend's
parallel-expression restriction. The follow-up uses the installed official
`mma_store_index_map` to express a lane/local epilogue, preserving numerical
order; it must be compiled and tested separately before promotion.

## Remote main audit during this experiment

Read-only review of origin/main 85bc044..7c868b1 found no new verified compliant
high-score source and no v731+ number collision. New submission_v470_merge is
AST-identical to the existing v470; the preliminary submission package contains
the existing v432. The old 84-point colleague implementation is still described
as using three Torch matmuls, not a new compliant candidate. Added old logs
mistakenly attribute 138992 to v478; retain the previously source-verified
138992=v496 mapping. Do not overwrite our longer optimization log with main's
historical edits or adopt its now-disproved '80 is unreachable' conclusion.

Current verified OJ baseline remains v718/v719/v720 at 80.33. Neither new probe
has an OJ submission. Compilation, correctness, and paired measurements below
must be backed by actual logs; this design is not a predicted score.

## Attempt 2: lane/local epilogue compiled and audited

The fallback uses the installed official 64-lane `mma_store_index_map` and a
64-element per-thread local FP32 accumulator. It avoids the rejected high-level
paired-fragment expression; Gate slots 0..15/32..47 pair with Up at +16.
No other mathematics, pass configuration, or synchronization was changed by
this fallback. Both K64 and K32 compiled successfully.

The **tested** Python source hashes are frozen here even if result-only header
comments are subsequently updated:

| Probe | Tested SHA256 |
| --- | --- |
| v731 | `f1271c866444c1b9921b721f47c4b6f7ba16b95e33f377669d210578296f4f5c` |
| v732 | `3bd4f4ce6c2216be2b69e334b76d9411820053e28668ce7c22f42c265cd14797` |

After measurement, only result comments and the final newline were updated.
The final repository files have the following distinct hashes; restoring those
non-executable changes reproduces the tested hashes above, with unchanged AST.

| Probe | Final documented-source SHA256 |
| --- | --- |
| v731 | `4f5f160cb39a4d9f91c6f61f877a9ea2fa2808a24dbe3682ca8bbde075ba66f5` |
| v732 | `dfff09677c0159beef0f7b3a2d75f0ba2921a388d80a8392499f3d2610c66d12` |

The captured source is in `codex_e32_731_732_codegen_attempt2.log` (SHA256
`41f93456341d9a22d696edd075fbd81d29b0c75ee455f652e3d7e96953a3c121`).
See [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md) for actual C++ address replay and
synchronization review. Both captures have correct producer/read and
read/overwrite barriers, in-range current-input copies, correct C-pair mapping,
and complete K16 coverage. This is not inferred merely from the Python source.

`audit_v731_v732_cpu.py` also passes Python/source-AST/geometry and host-only
mock checks: only target E32/H7168/I2048 dispatch changes, all original builders,
Stage2 and host paths remain unchanged. E1/E8/E16/E32/E64 with FP16/FP32 routing
and two fresh input sets launch two kernels per call; non-target E32 shapes
retain their original fallback. Reproduce from the repository root with:

```text
python xpuoj_data/bench_records/v731_v732/audit_v731_v732_cpu.py
```

## Untraced random input: correctness and all four entry samples

Raw log: `codex_e32_720_731_732_random_entry.log`, SHA256
`92281fd270904e975e5bbe1ce009623e19454854df27037d1e8c437aafdff455`.
This is local E32, H7168/I2048, alternating64-220 routing, 4544 raw rows,
6144 padded rows, 48 M blocks, FP16 inputs/matrix weights and FP32 route weights.
The helper's case-2 seed is 20260903. It is a local fixture, not verified OJ data.

For each of three correctness repetitions, both `launch_full` and the real
`run_kernel` entry recomputed after workspace/output were filled with NaNs.
All three versions reported `max_abs=0.000000`, `bad=0/44040192` on every
comparison; the reference was freshly computed c0/v720 output. These are three
recomputations of one fixed random input set, **not three independent seeds**,
not NaN inputs, and not an independent mathematical reference implementation.

The following times are for untraced `run_kernel` entry, one warmup call and
one timed call per round. Each entry still launches Stage1 and Stage2. The four
rounds alternate forward/reverse candidate order. All times below are ms.

| Round | Order | v720 | v731, K64 | v732, K32 |
| --- | --- | ---: | ---: | ---: |
| 1 | forward | 4.626688 | 7.034624 | 15.493632 |
| 2 | reverse | 4.678656 | 6.867968 | 15.469312 |
| 3 | forward | 4.660224 | 6.919680 | 15.427072 |
| 4 | reverse | 4.624640 | 6.877952 | 15.400192 |
| Median | | 4.643456 | 6.898816 | 15.448192 |
| Mean | | 4.647552 | 6.925056 | 15.447552 |
| Population stdev | | 0.022849 | 0.066166 | 0.036260 |

Same-round candidate-minus-v720 deltas (positive means slower), in microseconds:

| Candidate | Round deltas | Median delta | Mean delta | Wins / ties / losses |
| --- | --- | ---: | ---: | --- |
| v731 | +2407.936, +2189.312, +2259.456, +2253.312 | +2256.384 | +2277.504 | 0 / 0 / 4 |
| v732 | +10866.944, +10790.656, +10766.848, +10775.552 | +10783.104 | +10800.000 | 0 / 0 / 4 |

Median entry latency increased by 48.571% / 232.687%. The helper's printed
`speedup_vs_c0=-32.692%/-69.942%` uses `base/candidate-1`, a different denominator;
do not mislabel those numbers as percentage latency increases.

## Decision and evidence limits

**Do not recommend v731 or v732 for OJ submission.** Both are correct against
the local v720 reference but substantially slower in every measured round.
No OJ result exists for either. Keep v720 as the verified 80.33-point baseline.

The generated small LDS/MMA/epilogue loops lack explicit unroll pragmas and use
scalar accesses, but this does not prove that the native compiler leaves them
rolled or that they caused the regression. v733/v734 are separate loop-kind
isolation experiments; their results must not be credited to these source hashes.
Shared-memory reduction, logical accumulator size and reported register counts
do not by themselves establish fewer spills or higher active occupancy.
See [PROFILING.md](PROFILING.md) for the separate constant-input mcTracer run;
its times are not pooled with these untraced random samples.
