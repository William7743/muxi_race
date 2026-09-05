# v750 — E16 Stage2-only runtime M128/M64

**Not recommended for OJ.** CPU checks, E16 compilation/source audit and
two-fixture full/entry tolerance checks passed. First-fixture median improved
0.7046%, but the second regressed about2.01% and all4 paired calls were slower.
No OJ result exists. The source header preserves creation-time status and SHA;
later evidence is recorded below. Candidate authoring left parent probes intact.

## Frozen source and design

- Candidate: [probe_v750_v745_e16_stage2_runtime_m64.py](../../probe_v750_v745_e16_stage2_runtime_m64.py), SHA256 `9fe1ea13e226907071cffe3bdebf7e1c3f6d2f371b7818761c922fb79beeb207`.
- Parent: final [v745](../../probe_v745_v743_e32_stage1_runtime_m64.py), SHA256 `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465`.
- Independent of v749 and v748. All builders, Stage1 dispatch, existing caches/keys, mathematical operations, layouts, passes, tiles, and launch arguments are byte-for-byte unchanged from v745.
- Exactly four executable predicates change: Stage2 managed-shape selection, its inner runtime-shape selection, and the two host empty-input guards. Header comments are the only other changes.
- New target is **E16/H2048/I8192**. It reuses the existing `_moe_stage2_runtime_m64_route_bounds`: M128 for more than 64 actual rows, M64 for 1–64, full output zeroing for zero rows; one shared 128x64 Up and Down buffer, dual-B emitter, `k_pack=1`, swizzle 2. I8192 means 128 K64 tiles; no reduced K or hardcoded numeric result.
- Hypothesis: executing M64 on short expert tails may reduce Stage2 work. Long I8192 may change compilation/resources and the size of the benefit; this selector-only probe needs direct GPU evidence before any performance recommendation.

| Scope | v750 host/Stage2 behavior |
| --- | --- |
| Exact E16, padded rows = 0 | Return before workspace allocation or launch |
| Exact E16, positive padded rows, raw routes = 0 | Existing no-load zero-output Stage2 only; Stage1 skipped |
| Exact E16, positive raw/padded rows and block count | Original Stage1 plus existing runtime-M64 Stage2, two launches |
| Exact E16, positive raw/padded rows but zero block count | Existing clamped M128 fallback selected; no new validity guarantee for malformed block metadata |
| E32, including H2048/I8192 | All v745 behavior retained; H2048/I8192 does **not** newly select runtime-M64 |
| E16/H7168/I2048 or other nearby E16 shapes; E64 and other experts | All v745 paths and inherited boundary limitations retained |

No new async/BSM/pipeline, external device implementation, workspace, input modification, or result replay. Only compiled callable/workspace allocation caches are retained; normal calls invoke both stages with the current inputs. This does not assert all-shape memory safety.

## Reproducible CPU verification

[audit_v750_cpu.py](audit_v750_cpu.py) uses only the Python standard library and imports neither torch nor TileLang:

```text
python xpuoj_data/bench_records/v750/audit_v750_cpu.py
python -m py_compile xpuoj_data/probe_v750_v745_e16_stage2_runtime_m64.py
ruff check xpuoj_data/probe_v750_v745_e16_stage2_runtime_m64.py xpuoj_data/bench_records/v750/audit_v750_cpu.py
```

All passed. The audit reconstructs the full expected module AST with exactly the four predicate edits, independently reconstructs the complete executable source text, and checks every unmodified function's exact source. It also verifies both inherited emitters use k_pack1, swizzle2, generic K calculation, and no tensor load in the empty-output kernel.

The host mock covers **336 shape/count/dtype combinations × two fresh-input calls × v745/v750**: 14 expert/dimension tuples, raw counts 0/1/129, padded counts 0/256, block counts 0/2, FP16/FP32 routes. It verifies selection, requested 0/1/2 launches, one reusable workspace allocation, callable caching, argument identity from each new call, exact E16 Stage1 isolation, all non-target equivalence, and both cross-shape traps E32/H2048/I8192 and E16/H7168/I2048.

Mock zero-grid calls and deliberately inconsistent count combinations characterize host requests only. They do not establish GPU acceptance of invalid maps, zero-grid execution, bitwise numerical equality, or speed. Generic builder source equality also does not establish unchanged generated resources for this new E16 signature.

## Completed E16 validation and negative second fixture

[Actual FP32 codegen audit](CODEGEN_AUDIT.md) and
[raw source](codex_e16_750_stage2_codegen_fp32.log) verify128 K64 tiles,
full/tail LDS/address coverage and bounded raw loads. The [v751 boundary run](../v751/README.md)
exercises this same Stage2 selection/body, isolated and composed, with actual
bitwise comparisons for two fresh X/routes sets and both route dtypes.
That helper is not a standalone v750 normal-entry test; the following batch is.

Both fixtures have three NaN-poisoned full-chain/entry tolerance recomputations,
then warmup1/iters1/four F/R rounds. v750/v745 entry medians are
2.579584/2.597888ms on alternating routing (3/4 paired faster), but
2.590080/2.539008ms on synthetic routing (0/4 faster). Raw samples and
baseline-reference limitations are preserved in the v751 batch record.
Correctness passes do not establish an OJ speed gain; this probe stays isolated.
