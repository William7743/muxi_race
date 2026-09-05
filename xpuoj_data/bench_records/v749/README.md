# v749: E16 single-kernel M64/M128 Stage1 on v745

## Status and identity

**Not recommended for OJ: routing-dependent local regression.** Python/Ruff,
CPU/source checks, E16 compilation/source audit and two-fixture full/entry
tolerance checks passed. Entry median improved1.4486% in the first fixture but
regressed about1.24% in the second (only1/4 paired faster there). Preserve both
windows; no OJ result exists. Frozen header comments describe creation time.

- Candidate: [probe_v749_v745_e16_stage1_runtime_m64.py](../../probe_v749_v745_e16_stage1_runtime_m64.py).
- Frozen source SHA256: `1f057c8ee74f1385cb445a2bb8b9a3c89f6503522feba8e93d5a82c0ff853270`.
- Exact base: [probe_v745_v743_e32_stage1_runtime_m64.py](../../probe_v745_v743_e32_stage1_runtime_m64.py), SHA256 `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465`.
- CPU audit: [audit_v749_cpu.py](audit_v749_cpu.py), SHA256 `a8142517a3b44d60fa49862c284bec65ba5ec09f58efe991327f9a114e084b6d`.
- Version749 was unused in the local filename scan. This candidate deliberately does not inherit v748's unverified E64 OJ change.

## Exact change and hypothesis

Only E16/H2048/I8192 with positive padded-token and block counts selects the new `_moe_stage1_e16_runtime_m64_prefetch` builder. The actual E16 parent is `_moe_stage1_prefetch`, not the E32 GIU/terminal-K builder. Its full parameters, passes, copy widths, layout annotations, swizzle expression and reduction loop are preserved.

One external M128 `T.Kernel` keeps N128/K64 and256 threads. Two FP16 shared allocations remain A128x64 plus B128x64, totaling32KiB. A CTA-uniform branch chooses:

| Condition | Compute / accumulators | A shared argument | Workspace writes |
| --- | --- | --- | --- |
| actual_rows>64 | Original M128 Gate/Up fragments | Whole128x64 allocation | Valid rows only |
| 0<actual_rows<=64 | Independent M64 Gate/Up fragments | First64-row BufferRegion of the same allocation | Valid rows only |
| actual_rows=0 | No clears, loads or GEMMs | Unused | None |

The one128x64 Up-prefetch fragment is freshly overwritten on every current-K iteration in whichever branch is selected. No shared allocation, launch or global workspace is added. The proposed saving is fewer A rows loaded and fewer Gate/Up output rows computed in short blocks; the full branch still determines possible resource limits, so increased occupancy is not claimed.

The original definition remains outside the branch:

```python
active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)
```

Each branch executes the same original `range(active_k_steps)` loop, including the final iteration. For the target, K tiles are0..31 in order. Every K retains:

```text
Input -> A shared
Gate -> B shared (coalesced_width=4)
Up -> up_prefetch (coalesced_width=8)
Gate GEMM; explicit barrier
up_prefetch -> the same B shared (coalesced_width=4)
Up GEMM; explicit end-K barrier
```

There is no GIU reordering, terminal-K extraction or unrolling. `T.gemm` still uses Square policy and k_pack2; target swizzle is2 and shared vecSize is4. The FP32 SwiGLU expression and FP16 workspace output remain unchanged. The inherited predicated-load, safe-memory, vectorization and fast-math pass configurations are not altered.

The full branch's clearG/clearU/loop/valid-only epilogue is exactly the parent's AST. The tail changes only the two accumulator names, the A/epilogue row extent to64 and three A references: one copy destination and two GEMM arguments become `input_shared[0:tail_m, 0:bh1]`. The old inherited48KB comment is corrected to32KiB only in the new builder; original functions are not edited.

## Scope and preserved paths

Apart from the header, this file adds one builder and changes only the fallback leaf in `_get_stage1` to a nested exact-shape selection. E32, E64 and near-target E16 shapes keep their original choices; all original builders, Stage2, workspace/cache logic, `run_kernel` and other host helpers are text-identical to v745. Restoring the original getter and removing the added builder reconstructs the full parent module AST.

The getter does not introduce a new valid-token condition. Existing E16 host behavior is preserved even for zero valid tokens; zero-sized/nonpositive-block metadata uses the original builder selection. E32's existing zero-padded return and zero-valid output-zero kernel are unchanged. Positive normal inputs still invoke Stage1 then Stage2 with the current input objects. Allocation/JIT reuse never skips current-input computation.

No async/BSM/pipeline, external kernel, `T.import_source`, `T.call_extern`, numerical-result caching or benchmark-stage dispatch is introduced. The implementation agent did not run GPU/SSH/Git, alter submission.py, or edit the shared optimization log/queue.

## Completed reproducible CPU checks

From the repository root:

```text
python -m py_compile xpuoj_data/probe_v749_v745_e16_stage1_runtime_m64.py
ruff check xpuoj_data/probe_v749_v745_e16_stage1_runtime_m64.py xpuoj_data/bench_records/v749/audit_v749_cpu.py
python xpuoj_data/bench_records/v749/audit_v749_cpu.py
```

All passed. The first attempt used `python -m ruff`, which was unavailable in that Python environment; the installed `ruff` command passed without installing or modifying anything.

- Independently reconstructs the new builder from the actual parent and proves exact full-branch AST plus tail normalization with exactly three A views.
- Proves complete-module isolation and text identity of every original function except the narrowly changed getter; parameters, pass dictionaries, swizzle/layout and shared declarations are unchanged.
- Checks168 host metadata/dtype combinations, each with two fresh input sets: E1/E8/E16/E32/E64, two E16 near-target shapes, FP16/FP32 route dtypes, valid0/1/129, padded0/256 and blocks0/2. Exact target/fallback selection, current input identities, JIT/workspace reuse, two-launch paths and original E32 empty paths pass. Only the historical v743 audit's `Tensor` and `host_mock` definitions are AST-extracted; its top-level tests are not run.
- Executes the actual selected branch AST on symbolic shape/tag buffers for387 cases: rows0..128 crossed with Ksteps1/2/32. Every current-K Input/Gate/Up copy shape/tag, copy width, Gate-before-Up product, fragment selection and explicit shared-overwrite barrier passes. Positive blocks execute exactly two explicit barriers per K, including the last K; empty blocks perform no work.
- Enumerates every valid128-column workspace output once, verifies the unchanged scalar SwiGLU expression, and confirms no invalid/empty-row writes.
- Enumerates the declared shared vec4 layout: the full A128x64 footprint uniquely covers half offsets0..8191; its first64-row view covers0..4095 with the same64-half row stride and stays inside the32KiB shared budget.

## Required compiler / GPU follow-up and risks

CPU tag GEMMs are not real FP16/FP32 numerical MMA. The parent must inspect actual generated code before launches: first copy-to-GEMM RAW barriers, Gate-read-to-Up-overwrite protection, next-K overwrite protection, tail A-view lowering, independent accumulator layout and uniform branch control. Private-array declarations do not establish physical registers, spills or occupancy.

The branch may increase code size or resource pressure, while E16's larger intermediate dimension changes the balance of short-block savings and scheduling overhead. Successful E32 experiments do not prove E16 performance. The required local checks below now exist; OJ performance and stability remain untested.

## Completed local checks; no promotion

[Actual source audit](CODEGEN_AUDIT.md) and [raw generated source](codex_e16_745_749_stage1_codegen.log)
confirm E16 full/tail addresses and synchronization. Component boundary checks
in the [v751 batch](../v751/README.md) explicitly report bitwise-equal valid
Stage1 outputs and untouched NaN padding; they do not independently test
v749's whole positive entry. The separate batch does test that entry, three
same-input NaN-poisoned tolerance recomputations per fixture, with one warmup,
one timed call and four F/R rounds.

v749 versus v745 medians: alternating routing2.560256/2.597888ms,4/4 paired
faster; synthetic routing2.570496/2.539008ms,1/4 faster. These printed tolerance
checks are not bitwise comparisons. Keep the raw samples and reference caveats
in the batch record; do not merge this E16 path into the OJ baseline.
