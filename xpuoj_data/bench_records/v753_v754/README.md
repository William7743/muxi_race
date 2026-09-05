# v753 / v754: M32 Stage2 tail experiments on v748

## Status and frozen identities

Python syntax, Ruff and the complete CPU audit passed for both independent candidates. **v753 is stopped as an untested static control, not a GPU failure. v754 passed GPU boundary/entry correctness but is closed for promotion: its measured synthetic-routing entry median is0.4863% slower than v748 and all four paired rounds lose. No second fixture or OJ submission is pursued for v754.** Source headers point here for status and remain unchanged after testing.

| Source | SHA256 |
| --- | --- |
| [Exact v748 base](../../probe_v748_v747_e64_stage1_runtime_m64.py) | `af6b1c88d741d78de3b6a77a00d86afb136c39036c2bd76bea6daa361005ad20` |
| [v753: E64 M128/M32](../../probe_v753_v748_e64_stage2_runtime_m32.py) | `2aa038f9ab09e23252ebe6d9b77d969f0fd9269ac448edbba81a76aee350c517` |
| [v754: E32 M128/M64/M32](../../probe_v754_v748_e32_stage2_runtime_m32_m64.py) | `6d987d9ac8c4ac17dc900b6e186011ea957280ad70ba002c156efbb0ee14a909` |
| [Reproducible CPU audit](audit_v753_v754_cpu.py) | `372164a1596b8d2594f7c1df1259d1cdff02c8e31c1e3724d2cebfc0f837731e` |

The local candidate-filename scan found neither version occupied; an unrelated `oj_139753_verified.json` is a submission ID, not version753. No existing source or submission.py was replaced.

## v753: isolated E64 threshold change, stopped before GPU

v753 changes exactly three expressions inside the existing `_moe_stage2_runtime_m64_route_bounds` builder:

```python
tail_m = 32 if num_experts == 64 else 64
warp_row_tiles = tail_m // 2
for i, j in T.Parallel(bt1 - tail_m, bh2):
    out[block_start + tail_m + i, by * bh2 + j] = 0
```

The last expression replaces the old64-row zero loop; an M32 tail must zero rows32..127, not merely32..63. Existing dispatch confines the changed E64 use to H7168/I2048 with positive raw/padded/block counts. E64 rows33..128 use the original M128 body, rows1..32 use M32, and zero rows write the whole128-row output as zero. E32 retains64/32 tail/emitter dimensions; constant specialization of the new expressions reconstructs its original builder AST. All dispatchers, Stage1 and host source are untouched.

The initial E64 direction was halted after the parent's separate CPU expansion of the local synthetic routing fixture identified E64 tails40/41, which do not trigger M32. That is a distribution-based prioritization decision, not evidence of compile failure, incorrectness or measured slowdown. v753 was not GPU-tested and has no OJ result.

## v754: E32 three-way tail selection

The same local distribution investigation identified E32 tails27/28 as a more relevant M32 target. This is local fixture information, not official OJ routing data or a prediction of OJ speedup. v754 starts independently from v748, not v753, and adds `_moe_stage2_e32_runtime_m32_m64_route_bounds`.

| Uniform actual_rows condition | Executed geometry | Relationship to v748 |
| --- | --- | --- |
| >64 | M128/N128/K64 | Complete original M128 body unchanged |
| 33..64 | M64/N128/K64 | Complete original M64 body unchanged |
| 1..32 | M32/N128/K64 | Clone M64 body with independent tiny buffers/emitter and corrected zero extent |
| 0 | No GEMM | Original full128-row zero branch unchanged |

Only the existing positive exact E32/H7168/I2048 Stage2 selector chooses the added builder. E64/E16 and non-target E32 choose the original builders; original builder text, every Stage1 path, `run_kernel`, workspace/JIT-cache behavior and empty-input handling remain v748. The getter gains an E32-only conditional at the existing selected-builder leaf; its full allowed AST delta is audited.

M32 uses the official emitter with2 row warps ×2 column warps, warp tiles16×64, k_pack1, chunk64 and256 threads. It has independent tiny A/B0/B1 fragments and C32x128 layout. All paths share only the original A128x64 and B128x64 FP16 allocations:32KiB, not an extra shared tile. Tiny A reads/copies use the first32 rows; Down columns and K order are unchanged. M64 and M128 remain separate private layouts.

The dual-B K16 order, prologue, terminal K, explicit end-K barriers, inherited passes, swizzle2, vec4 layout, multiplication and raw-route clamp are unchanged. The tiny epilogue writes rows0..31 conditionally, then zeros32..127. No extra launch/global workspace, async/BSM/pipeline, extern, external numerical implementation or result cache is introduced. Positive normal inputs still compute Stage1 then Stage2 from current inputs.

## Completed CPU proof and its limits

Run `python xpuoj_data/bench_records/v753_v754/audit_v753_v754_cpu.py` from the repository root. The audit imports neither TileLang nor Torch and executes no GPU code. Both probes and the audit also passed the installed `ruff` command; the parent's `remote_v754_stage2_edges.py` independently passed Ruff without modifying that helper.

- v753: entire executable text and module AST differ by exactly the three expressions above; E32's compile-time constant specialization matches the original builder AST.
- v754: independently reconstruct the complete module from v748 plus declared tiny geometry, buffers/layout and branch; prove original M128/M64 bodies unchanged and every original function except the allowed getter text-identical.
- Both: exactly one Stage2 `T.Kernel`, exactly two shared allocations and every raw route subscript has the original clamp. Host216 metadata/dtype combinations each use two fresh input sets, covering experts1/8/16/32/64, near-target E32/E64 shapes, FP16/FP32 routes, positive/empty raw/padded/block metadata, unchanged launch arguments and empty shortcuts. Only historical host-mock definitions are reused, not their top-level tests.
- Actual branch AST runs774 row/K cases: rows0..128 × Ksteps1/2/32 for both changed designs. Check every copy row/K slice, current-K A/B tags, double-B lifetime, K16 order0/1/2/3 and explicit shared-overwrite barriers. This is symbolic execution, not numerical MMA.
- Actual epilogue AST runs516 row/dtype cases. Every128x128 output coordinate is written exactly once; all invalid rows are zero; route reads occur only for valid raw indices. Clamps are also checked for final and nonfinal raw ranges. Empty blocks perform no raw reads.
- Declared M32/M64/M1282×2-warp C ownership is one-to-one, and vec4 A-prefix footprints uniquely cover32/64/128 rows inside the same allocation.

CPU checks cannot prove actual TileLang M32 view lowering, automatic copy-to-read barriers, emitted register placement/spills, numerical tolerance or performance. The parent must inspect generated source and test31/32/33, empty/last-token boundaries and both route dtypes before benchmark or OJ. Adding a third branch may increase code/resource costs, and fewer computed rows do not imply greater occupancy or faster execution. No GPU, SSH, Git, shared optimization-log or queue mutation was performed by the candidate-authoring subtask.

## v754 completed GPU result: correct, no measured performance gain

[Boundary/codegen log](codex_v754_e32_edges_codegen_profile.log), SHA256 `40ece06b5da365e8341619ab2da71144dd72bdfc74cb21e7761fafda29ff4636`, confirms the frozen identities above. E32/H7168/I2048, raw2373/padded4608,36 blocks include31/32/33 and zero/final-token boundaries. Both FP32/FP16 route dtypes and fresh input seeds75401/75402 give four explicit `bitwise_equal=True`, finite=True, max_abs_full=0, nonzero_diff=0, padding_nonzero=0 checks. This is Stage2-only comparison against v748, not independent mathematical/OJ validation. See [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md) for the separate generated-source audit.

[Complete entry log](codex_e32_748_754_synthetic_random_entry.log), SHA256 `3cb237671b8ff1fd75a6b13fb98a6cd6cdefcd009161ae55a070a06a54ad0682`, reaches `stage_end stage=entry`. Synthetic routing/random values, E32/H7168/I2048, padded6912; both full-chain and actual `run_kernel` paths pass three same-input recomputations against saved `/root/ref_v432_case2.pt`:12 tolerance checks print max_abs=0.000000, bad=0/49545216. These are not three independent seeds or entry bitwise tests.

Untraced warmup1/iters1, four F/R/F/R rounds. Samples are ms:

| Version | Round1 | Round2 | Round3 | Round4 | Median | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v748 | 5.102336 | 5.083904 | 5.076992 | 5.076992 | 5.080448 | 5.085056 |
| v754 | 5.106432 | 5.103104 | 5.103872 | 5.172224 | 5.105152 | 5.121408 |

Candidate-minus-v748 paired deltas are+4.096/+19.200/+26.880/+95.232us; median+23.040us, mean+36.352us,0 faster/0 tied/4 slower. Median latency increases0.4863% (`100*(candidate/base-1)`), not the reciprocal speedup metric. The last sample is retained. Close v754 promotion and do not run a second fixture; fewer nominal tail operations did not yield a measured gain here. The final separate Stage1 experiment is v755, based on v748 and not combined with v754; no later optimization versions are planned at the user's request.
