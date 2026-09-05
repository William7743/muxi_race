# v755: final E32 Stage1 M32/M64/M128 experiment on v748

## Status and identity

This is the user's requested final optimization experiment. Python/Ruff/CPU checks, six GPU boundary bitwise comparisons and both normal-entry correctness windows passed. Synthetic routing with tiny tails is0.9641% faster by median latency, with4/4 paired wins; alternating routing with no tiny tails is0.0490% slower by median, with1/4 paired wins. **v755 is a final optional manual OJ candidate, not a proven replacement for the Accepted80.33 v748 baseline.** No v755 OJ ID/score exists. GPU work has ended and its lock is released. Per the user's request, optimization stops here: no v756+ or further exploration.

- [Frozen v755 source](../../probe_v755_v748_e32_stage1_runtime_m32_m64.py): `0b19e84d1695a16bca424ad7fd91f3a51b8baeb4f9e8b3cbf2fc3501224f94de`.
- [Exact v748 base](../../probe_v748_v747_e64_stage1_runtime_m64.py): `af6b1c88d741d78de3b6a77a00d86afb136c39036c2bd76bea6daa361005ad20`.
- [CPU audit](audit_v755_cpu.py): `7171fdb1a28c7141cb9f5ad35dae02e832b0145939819b6c5c3f270322f5c4e6`.
- v755 was unused in the local filename scan. Source headers point to this record for status and remain frozen. v754's negative Stage2 change is not included.

## Isolated design

Add one independent `_moe_stage1_e32_runtime_m32_m64_giu_merge` builder by cloning the actual v748 `_moe_stage1_runtime_m64_giu_merge`. Only its positive exact E32/H7168/I2048 getter leaf selects the new builder; E64, E16, near-target shapes and all Stage2/host behavior retain v748 source.

| CTA-uniform rows | Stage1 path | Change |
| --- | --- | --- |
| >64 | M128/N128/K64 | Original full body unchanged |
| 33..64 | M64/N128/K64 | Original tail body unchanged |
| 1..32 | M32/N128/K64 | Tail body with only `tail_` names changed to `tiny_`, and tiny_m32 |
| 0 | No workspace writes | Original empty-block behavior |

The new branch has separate tiny Gate/Up FP32 fragments. It uses the first32-row view of the original A128x64 shared allocation, the same B128x64 shared allocation, and the same freshly reloaded current-K Up-prefetch fragment. Shared stays32KiB; no global workspace or kernel launch is added.

All paths retain T.gemm/Square,256 threads,k_pack2, original pass dictionaries, shared vec4 layout and target swizzle3. Gate→Bshared, Input→Ashared, Up→prefetch, Gate GEMM, barrier, prefetch→Bshared, Up GEMM and steady end-K barrier remain in their original order. For H7168/K64, steady K0..110 precedes unchanged terminal K111; terminal has no new end barrier. FP32 SwiGLU order and valid-row-only FP16 workspace writes remain identical. No raw route access is added to Stage1.

No async/BSM/pipeline, external implementation, cross-call result reuse, loop change or compiler-flag change is introduced. The Stage2 experiment v754 is separately closed after a local slowdown; this source does not combine it.

## Completed CPU checks

Run `python xpuoj_data/bench_records/v755/audit_v755_cpu.py` from the repository root. Both source/audit passed `ruff check`; source Python compilation passed. The audit imports neither Torch nor TileLang and executes no GPU work.

- Independently reconstruct the full module AST from v748 plus two tiny accumulators, tiny_m32, strict tail renaming and the E32-only getter conditional. Original M128/M64 branch ASTs are unchanged; every original function except the permitted getter is text-identical, including Stage2 and `run_kernel`.
- Host216 metadata/dtype combinations each use two fresh input sets: experts1/8/16/32/64, near-target E32/E64 shapes, route FP16/FP32, raw/padded/block empty conditions. Verify exact target/fallback choice, current input arguments, original empty shortcuts and two-kernel paths. Only historical host-mock definitions are reused, not their old top-level tests.
- Execute the actual new branch AST on shape/tag buffers for387 cases: rows0..128 × Ksteps1/2/112. Verify GIU copy widths/slices, freshly overwritten Up prefetch, selected C fragments, matching Gate/Up current-K order, terminal-K coverage, and exactly2*Ksteps-1 explicit barriers for positive blocks. Tiny mock geometry is32; old tail/full geometry remains64/128.
- Every valid128-column workspace coordinate is written exactly once with the unchanged scalar SwiGLU expression. Empty and invalid rows stay untouched. Declared vec4 A-prefix footprints for32/64/128 rows uniquely cover the corresponding allocation prefix.

These symbolic checks do not prove GPU rounding, actual M32 T.gemm layout/view lowering, compiler-generated RAW barriers, physical register allocation/spills or speed. The parent must validate generated code and real boundary/entry behavior. Smaller tails may reduce work but add code/resource overhead; no occupancy or performance improvement is presumed. Source-authoring did not modify submission.py, parent probes, shared logs/queue or Git/GPU/SSH state.

## Completed GPU boundaries and first entry window

The [independent generated-source audit](CODEGEN_AUDIT.md) found no new bounded
address, K-order, padding-store or synchronization defect. Existing M128/M64
bodies remain intact; M32 A-prefix/C ownership, current-Up prefetch, terminal111
and valid-row-only SwiGLU stores were checked against the captured source.

The following artifacts retain the frozen v748/v755 source identities listed above:

| Artifact | SHA256 |
| --- | --- |
| [Boundary/codegen log](codex_v755_e32_edges_codegen_profile.log) | `5b6e0ac1b079179d394d81b00ef29520e35e6f1887dc3254ce3e27149809f79b` |
| [Boundary trace](edges-1123345.json) | `f9e55c69aec10bd59a2763c17a665e5792f51acf680b8715ff87e9fe8c766447` |
| [Complete synthetic entry log](codex_e32_748_755_synthetic_random_entry.log) | `37f3d44ed4ad1ca756883aed5c75bb738a42dc8f046c76f3eb6b745f80cbc855` |

Boundary geometry is E32/H7168/I2048, raw2373/padded4608,36 M blocks including0/31/32/33/63/64/65/127/128 and final-token cases. Fixed Gate/Up/Down seeds75510/75511/75512 accompany two fresh input seeds75501/75502. Two Stage1 valid-workspace comparisons each cover4859904 values; four current-workspace chains (two inputs × FP32/FP16 route dtypes) each cover33030144 outputs. **All six explicitly report finite=True, max_abs_full=0, nonzero_diff=0, bitwise_equal=True and tolerance_bad=0.** Both Stage1 `padding_untouched_nan=True`; all four chain `zero_padding=True`. These direct Stage1/Stage2 comparisons use v748, not an independent mathematical/OJ reference or positive `run_kernel` timing.

The trace has exactly12 matching `pid=2, ph="X"` events named `stage{1,2}_v755_edge_{base,probe}_kernel`; timestamps and durations are integer nanoseconds. Direct metadata is:

| Group | co_id values | registers_per_thread | dynamic_shared / static_shared bytes | private_per_thread / private_total |
| --- | --- | ---: | --- | --- |
| Stage1 v748 | 187,1991 | 248 | 32768 / 0 | 0 / 0 |
| Stage1 v755 | 195,1997 | 248 | 32768 / 0 | 0 / 0 |
| Stage2 v748, both route dtypes | 818,1416,2591,3164 | 154 | 32768 / 0 | 0 / 0 |
| Stage2 v755, both route dtypes | 826,1424,2597,3170 | 154 | 32768 / 0 | 0 / 0 |

Cold boundary durations/gaps are not used as performance samples. Equal resource fields do not prove equal occupancy or absence of spill/local-memory traffic; no bandwidth/stall counters or source-array-to-physical-register inference is made.

The separate first normal-entry window uses synthetic routing with random values, E32/H7168/I2048, padded6912, c0=v748/c1=v755. Its three same-input NaN-poisoned recomputations of `launch_full` and actual `run_kernel` produce12 tolerance checks with printed max_abs=0.000000, bad=0/49545216 against saved `/root/ref_v432_case2.pt`. This is not three independent seeds, entry bitwise equality or an independent OJ oracle. The complete log reaches `stage_end stage=entry`.

Untraced warmup1, one entry call/sample, four F/R/F/R rounds; each normal entry still launches both kernels. All chronological samples are ms:

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v748 | 5.115136 | 5.084416 | 5.070592 | 5.085696 | 5.085056 | 5.088960 |
| v755 | 5.003264 | 5.035776 | 5.056256 | 5.036288 | 5.036032 | 5.032896 |

v755-minus-v748 paired deltas are-111.872/-48.640/-14.336/-49.408us; median-49.024us, mean-56.064us,4 faster/0 tied/0 slower. Median latency reduction is0.96408% using `100*(1-candidate/base)`, not reciprocal speedup. This first-window positive signal does not by itself prove an OJ gain. The completed second window is below; no unfinished samples, cold trace times or assumed OJ score are included.

## Final second window and handoff

[Complete alternating-routing entry log](codex_e32_748_755_alternating_random_entry.log),
SHA256 `34b3114bb97c4afe3a97d7e6f293a1d0e8fe85ad1162de3b1c6eadf780385aca`,
reaches `stage_end stage=entry`. E32/H7168/I2048, raw4544/padded6144/48 CTAs,
alternating64-220 routing with random values has no positive <=32-row CTA.
Reference is freshly materialized from c0=v748, not the first window's saved
v432 tensor or an independent OJ oracle. All12 full/entry tolerance checks
across three same-input NaN-poisoned recomputations pass (bad0/44040192).

Untraced warmup1/iters1/four F/R rounds, identical candidate order748/755:

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v748 | 4.448000 | 4.435712 | 4.463360 | 4.434688 | 4.441856 | 4.445440 |
| v755 | 4.468224 | 4.443904 | 4.441856 | 4.444160 | 4.444032 | 4.449536 |

Candidate-minus-base deltas are+20.224/+8.192/-21.504/+9.472us;
median+8.832us, mean+4.096us,1 win/3 losses. The median ratio is+0.04899%
latency, a near-neutral observed difference, not proof of exact equality,
zero overhead or statistical equivalence. No samples or fixtures are pooled.
Only E32 is performance-tested in this final batch; E16/E64 isolation is
supported by complete source/host checks, not newly repeated numerical runs.

The tiny-tail signal supports **one optional final user-run OJ test of v755**.
It does not establish broad superiority, a score increase, or a stable win on
the full OJ device (the local device is a25% C500 slice). If no further OJ test
is desired, use frozen v748 with its existing140335 Accepted80.33 feedback.
The test process exited; at2026-09-05 14:05:13UTC no GPU process was listed,
slice usage was0/16000MiB and0%, and the owned GPU lock was released.
No candidate code is modified after its recorded SHA; submission.py is untouched.
