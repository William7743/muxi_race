# v745: single-kernel M64/M128 Stage1 on v743

## Status and identity

**Latest screenshot: OJ140316 Accepted,80; its association with the requested unchanged-v745 repeat is inferred from reply context, not explicit version text or uploaded-source verification. The earlier OJ140296 score72.67 regression did not recur in this repeat. Point2 is69us faster than the nearby v743/140309 repeat, but total score is unchanged: stable superiority or a score gain is not established. Python/Ruff, source-isolation, CPU symbolic/host checks, TileLang compilation and local Stage1/chain checks passed.**

- Candidate: [probe_v745_v743_e32_stage1_runtime_m64.py](../../probe_v745_v743_e32_stage1_runtime_m64.py).
- GPU-tested candidate SHA256: `12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96`.
- Final published-header SHA256: `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465`.
- Exact parent: [probe_v743_v723_e32_stage2_runtime_m64.py](../../probe_v743_v723_e32_stage2_runtime_m64.py).
- Parent SHA256 at cloning: `5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9`.
- Source was frozen before compilation and all GPU checks. Afterward, the parent replaced the one static-candidate comment with three result comments. Reversing that header replacement reproduces the tested SHA above; complete executable ASTs remain identical. Raw logs retain their original tested identity.
- This probe and its local audit/record files were authored for the implementation task; submission.py, parent probes, OPTIMIZATION_LOG, GPU/SSH and Git were not modified by the implementation agent.

## Hypothesis and exact design

Keep v743's Stage2 runtime-M64/clamped implementation and all other paths. Only E32/H7168/I2048 Stage1 gains a CTA-uniform M128/M64 choice inside one existing-size kernel launch. The intended saving is fewer A rows and fewer Gate/Up MMA rows for blocks with at most64 valid rows. Gate/Up weight columns and their global reads are unchanged. This is not a claim of occupancy improvement.

The actual parent Stage1 dispatcher selects `_moe_stage1_prefetch_giu_merge` (terminal-K), **not** the retained `_moe_stage1_prefetch_giu_merge_v527`. The new builder is `_moe_stage1_runtime_m64_giu_merge`.

| Component | Full branch | Tail branch |
| --- | --- | --- |
| Uniform condition | actual_rows>64 | 0<actual_rows<=64 |
| Logical GEMM | M128/N128/K64 | M64/N128/K64 |
| Threads | 256 | 256 |
| A shared | whole128x64 allocation | first64x64 BufferRegion of the same allocation |
| Reused B shared | same128x64 allocation | same128x64 allocation |
| Gate/Up C | separate128x128 FP32 fragments | separate64x128 FP32 fragments |
| Up prefetch | shared declaration128x64 FP16 fragment; newly loaded each K | same declaration, newly loaded each K |
| Math/API | T.gemm, Square policy, k_pack=2 | same |
| Workspace writes | only valid rows | only valid rows |

Exactly two shared allocations remain, A128x64 and B128x64 FP16: a32KiB budget. The tail does not allocate a separate A shared tile and cannot shrink the whole kernel's allocation to24KiB. Full and tail Gate/Up accumulators have different fragment buffers; no buffer is given two dynamic C layouts.

Both branches retain the parent's current-K sequence:

```text
Gate -> B shared; Input -> A shared; Up -> up_prefetch
Gate GEMM; explicit barrier
up_prefetch -> same B shared; Up GEMM; explicit end-K barrier
```

Steady K is0..110 for H7168/K64, followed by the original terminal K111. The terminal preserves Gate, Input, Up prefetch, Gate GEMM, Gate-to-Up barrier, Up copy, Up GEMM, with no added terminal end barrier. SwiGLU order remains `up * (gate * (1 / (1 + exp2(-gate * 1.44269504))))`, then the original FP16 workspace store. Swizzle3, vecSize4, passes, GIU copy widths and all math are unchanged.

The full branch includes the parent's complete clearG/clearU/positive-compute/epilogue AST. The tail is the same terminal-K body with only Gate/Up C names changed, the A/epilogue row extent reduced to64, and six A uses changed to first64-row BufferRegions (two copy destinations and four GEMM inputs).

actual_rows=0 does not clear or write workspace; no A/Gate/Up load or GEMM is executed. Invalid rows remain untouched, matching Stage1's existing contract. v743 Stage2 retains responsibility for final padded-output zeros.

## Minimal source delta and dispatch

Apart from the explanatory header, the parent file changes only by:

1. Adding the one new builder before `_pick_tiles`.
2. Selecting it in `_get_stage1` only within the E32 branch when H7168/I2048 and padded/blocks are positive:

```python
(
    _moe_stage1_runtime_m64_giu_merge
    if hidden == 7168 and intermediate == 2048
    and total_padded_tokens > 0 and num_blocks_m > 0
    else _moe_stage1_prefetch_giu_merge
)
if num_experts == 32
else ...  # original E64/fallback selection
```

The kernel-cache key, Stage2 dispatcher/builders, workspace allocation, run_kernel and all original functions except `_get_stage1` are byte-identical to the parent. E32 zero-padded early return and zero-valid Stage2-only zero kernel are inherited exactly. Positive inputs still launch exactly Stage1 then Stage2. Near-target shapes and nonpositive block-count cases keep the parent Stage1 selection.

No extra global workspace, external prepacking, launch, async/BSM/pipeline, extern/import_source, numerical result cache or benchmark-phase branch is introduced.

## Historical donor and limits of the negative evidence

The closest verified M64 geometry/GIU donor is `probe_v633_e32_stage1_m64_giu_tail.py::_moe_stage1_e64_tail64` (927–1065), selected for E32 by `_get_stage1_e32_split` (1533). Despite its legacy E64 name, it was tested as an E32 Stage1 tail: M64/N128/K64 at256 threads, two FP32 accumulators, synchronous GIU and k_pack2. It used its own24KiB shared allocation, swizzle2 and an unspecialized final K loop. v745 does not blindly import those scheduling differences: it preserves the current terminal-K and swizzle3.

v633 used two separate Stage1 launches followed by Stage2. OPTIMIZATION_LOG lines2822–2823 record Stage1/full3.149312/4.798080ms versus v6143.019904/4.683520ms, a regression. This is valid negative evidence for that split combination, but does not isolate launch cost, mixed-CTA scheduling, tail geometry or swizzle. v745 is not an already-established win merely because it removes the extra launch.

The local read-only scan of515 top-level Python files found no prior Stage1 PrimFunc with runtime M64/M128 compute on both sides of the same row-threshold branch; v743 Stage2 was the only corresponding single-kernel pattern. The historical split donor is not the same experiment.

## Completed CPU checks

- Reproducible CPU audit: [audit_v745_cpu.py](audit_v745_cpu.py), run with `python xpuoj_data/bench_records/v745/audit_v745_cpu.py` from the repository root, or by its absolute path from another directory. It imports no torch/TileLang and launches no GPU work. It reuses only the historical v743 audit's `Tensor`/`host_mock` definitions through AST extraction, without running that audit's top-level tests.
- Python compilation and Ruff passed on the exact frozen candidate. An initial Ruff invocation had a mistyped filename; rerunning against the correct candidate path passed.
- Whole-module AST isolation: removing the new builder and restoring the original `_get_stage1` gives exactly the v743 module AST.
- Every original function other than `_get_stage1` is also text-identical, not just AST-equivalent.
- Full branch is exactly the parent's final four statements: clearG, clearU, guarded computation, valid-only epilogue.
- Normalizing only the allowed tail names, M64 bounds and six A BufferRegions restores the full branch AST exactly.
- New-builder parameters, passes and layout annotations match the parent. Static allocation inspection confirms one T.Kernel, two shared allocations, one Up-prefetch declaration and four independent G/U C declarations.
- Reused the independent v743 host-mock function with v743/v745 ASTs:168 combinations covering E1/E8/E16/E32/E64, two E32 near-target shapes, FP16/FP32 routes, valid0/1/129, padded0/256 and blocks0/2. Each combination used two fresh sets of input objects. Target/fallback builder selection, current-input arguments, launch counts, zero cases and cache reuse without skipped computation all passed.
- Executed the actual branch AST on symbolic shape/tag buffers for387 cases: actual_rows0..128 crossed with Ksteps1/2/112. Verified every current-K Gate/Input/Up copy, matched Gate/Up GEMM K tags, selected-branch fragment initialization, Gate-before-Up order, and exactly2*Ksteps-1 explicit barriers for positive rows.
- Symbolic epilogue wrote each valid128-column row once, produced the unchanged scalar expression, and left all invalid/empty rows untouched. This is not a GPU FP32/FP16 rounding proof.
- Declared vec4 shared-layout enumeration covers offsets0..8191 uniquely for A128x64; the A64 view covers exactly0..4095 with the same stride64/swizzle and stays inside that allocation.

CPU symbolic GEMM records reduction-tile tags, not hardware MMA. These checks cannot establish TileLang view lowering, compiler-inserted RAW barriers, physical fragment placement, numerical tolerance or actual register usage.

## First compilation update

While CPU checks were finishing, the parent independently compiled the frozen candidate successfully. Raw artifact: [codex_e32_743_745_stage1_codegen.log](codex_e32_743_745_stage1_codegen.log).

| Generated Stage1 | Source SHA256 | Characters | Static synchronization sites |
| --- | --- | ---: | ---: |
| v743 | `32f3b1b3e9da925514a1aa02c6313d4bcd34adae330783fdc82e4f4a36f2692b` | 12945 | 9 |
| v745 | `67282b935424a980038cc9beed0de38dedf723278f7c1c8147645f9d69456782` | 25102 | 18 |

Raw log SHA256: `ed1eaeec132bd24de1039b605471c1feeae94906dcdd9c383b43455b38c26e57`.

Metadata retains B shared at byte0 and A shared at byte16384; the logical budget remains32KiB. Reported register/spill values are null, meaning unavailable. Eighteen is a static code-site count across both branches, not the number of barriers executed by every block.

Generated C++ addresses, fragment mapping and producer/consumer barriers are owned by the independent codegen audit; this compilation summary does not substitute for it. In particular, source-level separate C fragments do not prove that physical registers are reused efficiently: the combined kernel may inherit the full branch's resource limit or suffer extra compiler allocation/spills.

## GPU Stage1 and chained boundary checks

Raw evidence: [codex_v745_stage1_edges_profile.log](codex_v745_stage1_edges_profile.log), SHA256 `5bcd6d34924530001ef185af825c45b0049fed694738c7ea4e7e97591a3d01bb`. Detailed trace/resource extraction is in [PROFILING.md](PROFILING.md), kept separate from entry timing.

This uses synthetic E32/H7168/I2048, raw2373/padded4608 with36 M blocks, including an allocated empty block, rows0/1/63/64/65/127/128 and additional boundaries. Two fresh input seeds74501/74502 use fixed random Gate/Up/Down seeds74510/74511/74512. Invalid input rows, both workspaces and both final-output buffers are poisoned with NaN before their respective computations.

The remote v743 baseline printed in the log is the original tested-header SHA `67d25409b20cf6417d79375f57edb3770a79fb1a7a619eb8bc3ca9e3b6e0e7ec`, not the local published-header SHA `5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9` used at cloning. [The v743 identity record](../v743/README.md) documents exact old-header/hash restoration and unchanged complete AST. The tested candidate remains the frozen `12f9dcc1...` SHA above.

| Check | Seed | Route dtype | Compared elements | finite | max_abs / nonzero_diff | bitwise_equal | Padding result |
| --- | ---: | --- | ---: | --- | --- | --- | --- |
| Stage1 valid workspace | 74501 | Not applicable | 4859904 | True | 0 / 0 | True | Invalid rows remain NaN |
| Stage1 valid workspace | 74502 | Not applicable | 4859904 | True | 0 / 0 | True | Invalid rows remain NaN |
| Stage2 on current workspace | 74501 | FP32 | 33030144 | True | 0 / 0 | True | Zero |
| Stage2 on current workspace | 74501 | FP16 | 33030144 | True | 0 / 0 | True | Zero |
| Stage2 on current workspace | 74502 | FP32 | 33030144 | True | 0 / 0 | True | Zero |
| Stage2 on current workspace | 74502 | FP16 | 33030144 | True | 0 / 0 | True | Zero |

All tolerance_bad counts are0, and the final log reports `all_tested_bitwise=True`. Stage1's finite test gathers only valid rows: the entire padded workspace is intentionally not finite. Final-valid-token nonzero assertions also pass. These are direct Stage1/Stage2 baseline comparisons, not a normal-entry timer or independent OJ reference.

The accompanying trace reports Stage1 registers_per_thread248 and dynamic_shared32768 for both versions, and Stage2 registers_per_thread154/shared32768 for both route dtypes and both versions. Static shared and reported private fields are0. This shows no metadata growth in this diagnostic; it does not prove no spill traffic or improved occupancy. Cold diagnostic durations are not reused below.

## Untraced entry window 1: alternating64-220

Raw: [codex_e32_720_743_745_random_entry.log](codex_e32_720_743_745_random_entry.log), SHA256 `9869ae1aaf113500e9edef24aa51c6697dcd9bc92a6005c76282350464ad8f21`.

E32/H7168/I2048, raw4544/padded6144,48 M blocks, random values with routing `alternating64-220`. Candidate list is `c0=v720, c1=v743, c2=v745`; rounds alternate forward/reverse. One entry warmup and one entry call per timing sample, four rounds, no tracer. The helper's entry `launches=1` means one callable entry; the submission still launches Stage1 then Stage2 internally.

All candidates pass three repeated fresh-workspace/output-poisoned checks for both `launch_full` and `run_kernel`: printed max_abs=0.000000 and tolerance bad=0/44040192. These entry checks are tolerance checks, not bitwise-equality measurements. The repetitions use the same deterministic input fixture, not three independent random seeds, and compare with freshly recomputed v720.

All timing values are ms, in chronological round order:

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 4.686080 | 4.664832 | 4.663808 | 4.674560 | 4.669696 | 4.672320 |
| v743 / c1 | 4.578048 | 4.553728 | 4.573440 | 4.584192 | 4.575744 | 4.572352 |
| v745 / c2 | 4.430592 | 4.477952 | 4.478208 | 4.470528 | 4.474240 | 4.464320 |

Paired deltas use **v745 minus same-round control**, in microseconds; negative favors v745:

| Control | Four paired deltas (µs) | Median delta | Mean delta | Wins/ties/losses |
| --- | --- | ---: | ---: | --- |
| v720 | -255.488, -186.880, -185.600, -204.032 | -195.456 | -208.000 | 4/0/0 |
| v743 | -147.456, -75.776, -95.232, -113.664 | -104.448 | -108.032 | 4/0/0 |

By the separate median ratio, v745 latency is4.1856% lower than v720 and2.2183% lower than v743 in this window. These latency-reduction percentages differ from the helper's reciprocal `speedup_vs_c0` metric.

## Untraced entry window 2: same routing, swapped candidate positions

Raw: [codex_e32_720_745_743_repeat_random_entry.log](codex_e32_720_745_743_repeat_random_entry.log), SHA256 `e6b0ae72ef28ffa222996bc2510ae45b0e89647eba06b8af45797eb9f282b67b`.

Same deterministic input/routing fixture and one-warmup/one-call/four-round method. Candidate list changes to `c0=v720, c1=v745, c2=v743`; this reverses the candidate ordering within each forward/reverse round. It is a new timing window, **not** an independent random-data fixture. Three repeated `launch_full`/`run_kernel` tolerance checks again pass with printed max_abs=0.000000, bad=0/44040192; this is not a bitwise-equality measurement.

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 4.640512 | 4.646144 | 4.660992 | 4.684032 | 4.653568 | 4.657920 |
| v745 / c1 | 4.421376 | 4.492032 | 4.482048 | 4.472064 | 4.477056 | 4.466880 |
| v743 / c2 | 4.544256 | 4.531200 | 4.587776 | 4.562944 | 4.553600 | 4.556544 |

| Control | Four paired deltas (µs) | Median delta | Mean delta | Wins/ties/losses |
| --- | --- | ---: | ---: | --- |
| v720 | -219.136, -154.112, -178.944, -211.968 | -195.456 | -191.040 | 4/0/0 |
| v743 | -122.880, -39.168, -105.728, -90.880 | -98.304 | -89.664 | 4/0/0 |

Separate-median latency is3.7930% lower than v720 and1.6810% lower than v743. These two windows give8/8 paired wins versus each control on one routing fixture; the second fixture follows below.

## Untraced entry window 3: synthetic routing, random values

Raw: [codex_e32_720_745_743_synthetic_random_entry.log](codex_e32_720_745_743_synthetic_random_entry.log), SHA256 `dbfe611842a72217b8f740730a115e1386bd168d4271adf328738945b5101bb3`.

E32/H7168/I2048, raw4544/padded6912,54 M blocks. This is a different routing fixture with random values; the list remains `c0=v720, c1=v745, c2=v743`, with the same one-warmup/one-call/four-round untraced method. All candidates pass three repeated NaN-poisoned `launch_full`/`run_kernel` tolerance checks: printed max_abs=0.000000, bad=0/49545216. These checks do not report bitwise equality. Unlike the first two windows' freshly computed c0 reference, this diagnostic uses saved `/root/ref_v432_case2.pt`; it is a local baseline reference, not an independent mathematical or OJ oracle. The candidate does not use that reference or cache results.

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 5.166080 | 5.229824 | 5.185280 | 5.220352 | 5.202816 | 5.200384 |
| v745 / c1 | 5.092608 | 5.091584 | 5.069568 | 5.147136 | 5.092096 | 5.100224 |
| v743 / c2 | 5.184000 | 5.160960 | 5.180416 | 5.166592 | 5.173504 | 5.172992 |

Times are ms. Paired deltas remain v745 minus same-round control, in microseconds:

| Control | Four paired deltas (µs) | Median delta | Mean delta | Wins/ties/losses |
| --- | --- | ---: | ---: | --- |
| v720 | -73.472, -138.240, -115.712, -73.216 | -94.592 | -100.160 | 4/0/0 |
| v743 | -91.392, -69.376, -110.848, -19.456 | -80.384 | -72.768 | 4/0/0 |

Separate-median latency is2.1281% lower than v720 and1.5736% lower than v743 in this fixture.

## Completed local conclusion

Across two routing fixtures and three timing windows, v745 is faster in all12 same-round pairs against each control. Separate-median latency reductions span2.13%–4.19% against v720 and1.57%–2.22% against v743. This was the historical basis for its first manual OJ submission; the subsequent OJ72.67 result below supersedes that recommendation. Reused deterministic inputs and repeated timing windows are not12 independent random fixtures or a statistical-significance proof; local results do not predict an OJ score.

All planned local checks in this batch are complete. The tested identity is preserved above; the published file differs only by result comments. Independent generated-source verification is in [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md). The persistent [CPU audit](audit_v745_cpu.py) SHA256 is `ceb7de0db54b51cba10b15396c4ffaa248e5a7c09d5f6619e348b720e734d267`. The separate [GPU diagnostic helper](../../remote_v745_stage1_edges.py) SHA256 is `5833ddc174d373fb3b796cfc73f6363fea3b89b7ad2757d8f5325188adffb382`.

The main thread confirmed all GPU jobs terminal and no benchmark processes remaining before releasing its GPU lock. Keep submission.py and the OJ-validated baselines unchanged. No gain is claimed for E16/E64 paths, which this candidate does not alter; local sliced-device timing does not guarantee full-device OJ gains.

## OJ140296 screenshot feedback (2026-09-05): promotion paused

The user identifies v745 as [submission140296](https://xpuoj.com/contest/5/submissions/140296)
and supplies a screenshot showing Accepted,72.67. Formal point1=3733us;
point2 displays7ms and point3 displays14ms (exact microseconds unknown).
Sample1=3747us is separate. Total25ms and memory22.3G retain UI precision;
individual point scores are not visible. [Structured transcription](oj_140296_user_report.json)
and [original screenshot](oj_140296_user_screenshot.png) preserve the evidence.

Compared with v743/OJ140270, the unchanged E16 point1 also slows2567→3733us
(+45.42%); unchanged E64 point3 goes from a rounded9ms to14ms. Thus the
cross-submission slowdown does not isolate the E32 Stage1 change, but does
not prove a platform/load issue either. Request a fresh manual v743 baseline
repeat in the current OJ window; do not promote v745 or infer its benefit
by rescaling these points. Source upload, runtime/device/load and timing
conditions remain unverified for this submission.

Uploaded source has not been retrieved/hashed. The delivered artifact is
commit b492df528e365561b3e4cd05702d4fac2355e3fc with the final published-header
SHA above. The version mapping is user-provided and result evidence is a
screenshot, not an authenticated API capture. No browser or new login was used.

The requested baseline repeat subsequently returned OJ140309,Accepted80,
2568us/4599us/rounded9ms (v743 association inferred from the reply context,
not uploaded-source verification). Its first two points match the previous
v743 timing range. A single unchanged-v745 repeat is now requested to check
reproducibility; this is diagnostic resubmission, not renewed promotion.

An independent read-only whole-module audit also confirms the E16/E64
reachable builders, arguments, host path, workspace/cache behavior and launch
order are unchanged (96 metadata/dtype combinations, two fresh inputs each).
There is one additional top-level `@tilelang.jit` wrapper construction for the
new E32 builder, not an additional top-level kernel launch. This source audit
does not verify OJ's uploaded file, import/JIT timing or evaluation conditions.

A pinned local TileLang0.1.10+maca compile-only check of E16/H2048/I8192,
padded3072/raw2272/blocks24 also generated identical complete device source
for v743/v745 after replacing only the diagnostic kernel name suffix1→0:
Stage1=9900 characters, Stage2(FP32 routes)=19169 characters. Raw record:
[codex_e16_743_745_unchanged_codegen.log](codex_e16_743_745_unchanged_codegen.log),
SHA256`2983a3b5377f39c61c416d2c63b24a72e09dc6aa18a40699db9d9b7d9ad12476`.
No kernels were launched in this check, no OJ binary was retrieved, and no
physical resource inference is made from source arrays or null accessors.

## OJ140316 screenshot repeat (2026-09-05): recovered to80, no score gain

The user's screenshot reply to the unchanged-v745 repeat request shows
[submission140316](https://xpuoj.com/contest/5/submissions/140316), Accepted80,
total16ms and memory22.3G. The v745 mapping is contextual, not explicitly
stated in the image or verified against uploaded source. Preserve this limit
as `version_mapping_explicit=false` and `uploaded_source_verified=false` in
the [structured transcription](oj_140316_user_report.json).
The [screenshot](oj_140316_user_screenshot.png) SHA256 is
`5e96a7aab36e1363c7aeda392834642213e983845297792ec34542873c9ef220`.

| Item | Status | Displayed time | Exact microseconds | Point score |
| --- | --- | --- | ---: | --- |
| Sample1 (not a scored point) | Accepted | 2565us | 2565 | Not applicable |
| Formal1.1 | Accepted | 2565us | 2565 | Unknown |
| Formal1.2 | Accepted | 4530us | 4530 | Unknown |
| Formal1.3 | Accepted | 9ms | Unknown | Unknown |

Against v743/140309 (also context-mapped), point2 decreases4599→4530us:
69us, or1.5003%; point1 decreases2568→2565us by3us. Both point3 displays
are rounded9ms, so no point3 difference, exact total, or individual point
score is inferred. Both overall scores are80.

The earlier72.67 severe slowdown did not recur in this repeat. This does
not establish its cause, stable v745 superiority, or a new score gain.
The failure record above is retained. The next separately selected candidate
is v748 for manual OJ testing; it is not an already-demonstrated OJ gain.
No API query, login, browser action or code resubmission was performed for
this screenshot transcription.
