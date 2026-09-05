# v725/v726: immediate current-K wide staging, rejected after random A/A/B checks

Date: 2026-09-05. Both are independent E32 Stage1-only changes from v720;
`submission.py` and all other shape paths remain unchanged. No OJ submission IDs.

- v725: Gate -> existing Up fragment at cw8 -> weight shared at cw4, then original Input/Up copies.
- v726: original Gate copy, then Input -> existing Up fragment at cw8 -> input shared at cw4, then original Up copy.
- Both consume staging immediately, allocate no additional buffer, preserve current-input recomputation,
  K order, all math and explicit synchronization. These are not cross-K prefetch or async copies.
- Generated source retains the intended `uint4` global read and `uint2` shared store. Total bytes do not
  decrease. Compiler-inserted staging barriers increase static sites from baseline summary 9 to 11;
  shared offsets remain 0/16384. Static source counts are not hardware instruction counts.

## Untraced random screening

Local quarter-C500 instance, TileLang 0.1.10+maca, E32/H7168/I2048, 6144 padded rows,
4544 valid rows, 48 M blocks; alternating64-220 is a local fixture, not verified OJ routing.
Seed 20260903. Three recomputations of the same random input, NaN-poisoned workspace/output,
both launch_full and real run_kernel: all four candidates exact against fresh candidate-zero output,
`max_abs=0, bad=0/44040192`. This is differential testing, not an independent mathematical oracle.

Warmup 1, iterations 1, rounds 4 alternating forward/reverse; no profiler in this run.

| Candidate | Median entry ms | Paired median delta vs c0, us | Wins/losses |
| --- | ---: | ---: | ---: |
| c0 v720 | 4.658944 | 0 | control |
| c1 v720 A/A | 4.649088 | -4.096 | 3/1 |
| c2 v725 | 4.918016 | +251.136 | 0/4 |
| c3 v726 | 4.992000 | +329.600 | 0/4 |

Relative median time increases: v725 +5.561%, v726 +7.149%. The duplicate baseline has a
different compiled module/name and workspace, so A/A is not solely a clock-noise measurement.
Both candidates lose to both baselines in every round. **Do not prioritize either for OJ.**
No second routing fixture is warranted by this negative first screen. No new score claim.

Raw samples and correctness: `codex_e32_720_720_725_726_random_entry.log`.
CPU audit: `python xpuoj_data/bench_records/v725_v726/audit_v725_v726_cpu.py`.
It checks complete-source/AST isolation, K=1/2/3/4/31/32/111/112/113 dependencies,
complete cw8->cw4 lane mapping, and E1/8/16/32/64 x FP16/FP32 dispatch with fresh inputs.
Python/Ruff passed. Pure paired-statistics harness tests: 7 passed.

## Actual profiling evidence

See [PROFILING.md](PROFILING.md). `mcTracer` now provides actual reported register/shared/private
metadata and stage timings, avoiding inference from source array declarations. v720/v724/v725/v726
Stage1 registers/thread are respectively 248/256/240/242. Lower registers did not make these wide
staging candidates faster. Added synchronization is a plausible contributor, not proven sole cause;
no memory-bandwidth, bank-conflict or instruction-stall counters were collected.

Raw captures `v720-1076736.json` and `candidates-1077079.json` have integer nanosecond timestamps.
Do not silently interpret them as ordinary Chrome trace microseconds or round absolute timestamps
through JavaScript floating point. Both diagnostic runs used constant input and are not precision
evidence or score measurements. Separate untraced random evidence is above.

## Source identity

Tested v725 SHA256: `fb290d2f99a09f4f692304779ec6e8a9d27ca6e5f0c16333460923291a86d984`.
Tested v726 SHA256: `f8bb125024b95dd98312b4d2aab878002eaab3e32290dac2206a0b50a01b624c`.
After tests, only result/header comments were updated (including distinguishing explicit from
compiler-inserted barriers); execution code is unchanged.
Final v725 SHA256: `51700a3f7560339d22fafbdbd7665751e8415b0a6d065ac4f3a21be7d7fac340`.
Final v726 SHA256: `d53e23cc62539c10657535b8be515d10ee8ef9606be02b5eaf61008747d83adf`.

GPU lock was released after both trace processes and the untraced benchmark finished.
