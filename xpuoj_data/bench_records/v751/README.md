# v751: combined E16 runtime-M64 stages on v745

## Status and identity

**Do not recommend v751 for OJ: the second routing fixture regresses by3.4231% median latency versus v745 and loses all four paired rounds, overturning the first window's2.2320% gain as a robust recommendation.** CPU composition, component code generation,10 boundary bitwise comparisons/empty-input checks and both windows'24 tolerance comparisons each passed. Correctness did not fail; cross-fixture performance did. v752's static combination is paused without CPU audit or GPU testing. No OJ result or gain is claimed for this E16 batch.

- [v751 source](../../probe_v751_v749_e16_stage2_runtime_m64.py): `084dbc599bb2bbe1f5b14dbf6527ba8ced2ce350d57169dc40abaac5fb51eb94`.
- Exact composition: v749's entire source, replacing only `_get_stage2` and `run_kernel` with v750's versions; explanatory header aside, all other executable text is unchanged. E16/H2048/I8192 receives both runtime-M64 stages and the Stage2 raw-route clamps/empty-input paths. E32/E64 paths are not expanded.
- [CPU audit](audit_v751_cpu.py): `0ec0a2f1e046cceca1ddab3df6b1b659638f00e7435a739967f8102273db3716`; parent-run complete AST/executable-text proof and216 metadata/dtype combinations with two fresh input sets each passed. Re-run with `python xpuoj_data/bench_records/v751/audit_v751_cpu.py`.
- The tested remote v745 baseline is `12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96`, not its later published-header `ec864ca3...` file; [v745's identity record](../v745/README.md) documents header-only restoration and unchanged executable AST.

## Completed edge diagnostic: what was actually compared

Evidence: [raw log](codex_v751_e16_edges_profile.log), SHA256 `cdf96a8a54692fe67fb9da44df465bd516d04e390c4ce8c6baae53d49a6085a5`; [trace](edges-1118046.json), SHA256 `d109d8dcbf40f5d774a19de37a2363d09dd45e41fb425778bff5c2a640a12c35`.
The [diagnostic helper](../../remote_v749_v751_e16_edges.py), SHA256 `c9113bb999c6f6050230fcd323be6457142374a404c2209276d979686a3806f4`, uses Torch only for external data generation/comparison, not candidate computation.

Local E16/H2048/I8192, raw1213/padded2432,19 M blocks. Row counts are `[0,1,15,16,31,32,63,64,65,95,127,128,128,1,128,63,128,127,1]`. Fixed Gate/Up/Down seeds75110/75111/75112 accompany two fresh X/routes batches75101/75102. Invalid X rows, Stage1 workspaces and each output start NaN.

Reference Stage1 is the original v745 E16 path. Reference Stage2 is explicitly `_moe_stage2_fast_bfrag_prefetch_route_bounds` at M128, **not** v745's normal E16 dispatcher. Isolated Stage2 uses the current baseline workspace on both sides; the chain then uses the current v751 workspace. Positive tests call the two stages directly, not `run_kernel`.

| Actual comparison | Count | Elements per comparison | finite / max_abs_full / nonzero_diff | Explicit bitwise_equal |
| --- | ---: | ---: | --- | --- |
| Stage1 valid workspace, two seeds | 2 | 9936896 | True / 0 / 0 | True for both |
| Isolated Stage2, two seeds × FP32/FP16 routes | 4 | 4980736 | True / 0 / 0 | True for all four |
| Current-workspace chain, two seeds × FP32/FP16 routes | 4 | 4980736 | True / 0 / 0 | True for all four |

All10 tolerance_bad counts are0, but bitwise equality is separately measured, not inferred from tolerance. Both Stage1 `padding_untouched_nan=True`; the finite check gathers valid rows only. All four chain `zero_padding=True` checks cover both isolated and chained outputs, and final-valid-token nonzero assertions pass. Final `all_tested_bitwise=True` is printed.

For both route dtypes, actual empty-route `run_kernel` calls produce finite all-zero outputs while a prohibitor replaces the Stage1 getter; it is never called. Both zero-padded-output calls return with Stage1/workspace getters prohibited, proving those measured host shortcuts. These empty calls are distinct from the direct positive-stage diagnostic.

v751 shares Stage1 with v749 and Stage2/host selection with v750. This establishes the components exercised here; it does not mean the complete v749 and v750 modules have each already passed numerical GPU/normal-entry tests. The comparison is against existing local implementations, not an independent mathematical or OJ oracle.

## Actual mcTracer events and resource metadata

Select `pid=2, ph="X"` names `stage{1,2}_v751_edge_{base,probe}_kernel`, sorted by `ts`:18 events. All timestamps and durations are integer **nanoseconds**; for example first `ts=1788612966753190144`, `dur=1259776`. Labels follow the fully read helper/log order. These are cold boundary calls with compilation/comparison activity, not warmed performance samples; neither duration ratios nor gaps establish stable speedup, idle time or host overhead.

| Operation / role | Seed75101: co_id / dur(ns) | Seed75102: co_id / dur(ns) |
| --- | --- | --- |
| Stage1 baseline | 189 / 1259776 | 2984 / 1278976 |
| Stage1 v751 | 197 / 1081344 | 2990 / 1091328 |
| Stage2 FP32 reference M128 | 836 / 858112 | 3588 / 857088 |
| Stage2 FP32 v751, baseline workspace | 844 / 777984 | 3594 / 777472 |
| Stage2 FP32 v751, v751 workspace | 850 / 771840 | 3600 / 797184 |
| Stage2 FP16 reference M128 | 1913 / 849664 | 4661 / 849152 |
| Stage2 FP16 v751, baseline workspace | 1921 / 782080 | 4667 / 820736 |
| Stage2 FP16 v751, v751 workspace | 1927 / 772608 | 4673 / 789504 |
| Empty-route zero kernel, FP32 (after both seeds) | 5758 / 8448 | Not repeated per seed |
| Empty-route zero kernel, FP16 (after both seeds) | 5892 / 7680 | Not repeated per seed |

Direct `args.mem` values, identical within each group:

| Kernel group | registers_per_thread | dynamic_shared bytes | static_shared bytes | private_per_thread / private_total |
| --- | ---: | ---: | ---: | --- |
| Stage1 baseline and v751 | 228 | 32768 | 0 | 0 / 0 |
| Reference M128 Stage2, both route dtypes | 154 | 32768 | 0 | 0 / 0 |
| v751 positive Stage2, both dtypes/workspaces | 152 | 32768 | 0 | 0 / 0 |
| v751 empty-route zero kernel, both dtypes | 8 | 0 | 0 | 0 / 0 |

All blocks are256×1×1; Stage1 grid19×64×1, all Stage2 grids19×16×1. Every event reports device0, queue0, hw_queue_id `[1879053792]`, max_block_size256, is_dynamic_parallel=false and is_recompiled=false. Raw `mtreg_occupancy(%)` is44/30/29/1 for the four resource groups; raw `shared_memeory_occupancy(%)` is50/50/50/0. Preserve those field names: neither is measured SM-active occupancy or bandwidth.

Stage1 metadata is unchanged; positive Stage2 reports two fewer registers/thread than the explicitly clamped M128 reference. This alone proves no scheduling, occupancy or performance improvement. Private fields0 do not prove absence of spill/local-memory instructions or traffic; there are no bandwidth/cache/stall counters here. Source arrays are not physical-register evidence.

## Batch decision

Both normal-entry windows are complete below. The second synthetic-routing/random-value fixture, ordered745/751/750/749, is negative for all three candidates by median latency. Stop promotion of v751 and pause v752; preserve the first positive window as history, not the final recommendation. The parent has released the GPU lock and is instead requesting an existing v747 versus v748 OJ comparison to isolate E64 Stage1, not submission of this E16 batch.

## Untraced entry window1: alternating64-220

[Complete log](codex_e16_745_749_750_751_random_entry.log), SHA256 `b23c5f1cbc1e7a6df10f0d9872916ebc7598d85257e558d0aeb4d783f82a913c`, reaches `stage_end stage=entry`. E16/H2048/I8192, padded3072,24 M blocks, random values and alternating64-220 routing; list c0=v745, c1=v749, c2=v750, c3=v751. All four complete modules pass three repeated `launch_full` and `run_kernel` comparisons against c0 v745:24 checks print max_abs=0.000000 and tolerance bad=0/6291456. These are same-input recomputations, not three independent seeds; entry bitwise equality is not measured.

Each callable entry has one warmup, then one timed invocation in each of four forward/reverse/forward/reverse rounds, without tracing. The helper's `launches=1` denotes one entry call, which internally launches both kernels. Samples are chronological ms; latency reduction is100*(1-candidate_median/v745_median), not the reciprocal logged speedup.

| Version | Round1 | Round2 | Round3 | Round4 | Median ms | Reduction vs v745 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v745 / c0 | 2.612736 | 2.619904 | 2.577920 | 2.583040 | 2.597888 | 0.0000% |
| v749 / c1, Stage1 only | 2.561280 | 2.598656 | 2.559232 | 2.524928 | 2.560256 | 1.4486% |
| v750 / c2, Stage2 only | 2.575104 | 2.601216 | 2.557184 | 2.584064 | 2.579584 | 0.7046% |
| v751 / c3, combined | 2.535168 | 2.544640 | 2.523136 | 2.550784 | 2.539904 | 2.2320% |

Same-round deltas are candidate minus v745 in microseconds; negative favors the candidate:

| Version | Four deltas us | Median delta us | Mean delta us | Faster / tied / slower |
| --- | --- | ---: | ---: | --- |
| v749 | -51.456, -21.248, -18.688, -58.112 | -36.352 | -37.376 | 4 / 0 / 0 |
| v750 | -37.632, -18.688, -20.736, +1.024 | -19.712 | -19.008 | 3 / 0 / 1 |
| v751 | -77.568, -75.264, -54.784, -32.256 | -65.024 | -59.968 | 4 / 0 / 0 |

The combined version is locally positive in this first window, while Stage2-only loses one round. The completed second window below rejects cross-fixture promotion. No additive causal decomposition, statistical significance or OJ score gain is established. Cold trace durations are not pooled into either entry table.

## Untraced entry window2: synthetic routing, negative result

[Complete log](codex_e16_745_751_750_749_synthetic_random_entry.log), SHA256 `adea18340c09013f6459b6e70794902f372a98103954a4239986d5fd2c2bbaa7`, reaches `stage_end stage=entry`. E16/H2048/I8192, padded3072, synthetic routing and random values; order c0=v745, c1=v751, c2=v750, c3=v749. All24 three-repeat full/actual-`run_kernel` comparisons print max_abs=0.000000 and tolerance bad=0/6291456 against saved `/root/ref_v432_case1.pt`, unlike window1's c0 reference. These are same-batch recomputations, not three independent seeds or entry bitwise proofs; the saved reference is diagnostic-only, not used by candidates.

Same untraced one-warmup/one-entry-call method, four F/R/F/R rounds. All chronological samples are retained in ms. Reduction remains100*(1-candidate_median/v745_median); negative means slower.

| Version | Round1 | Round2 | Round3 | Round4 | Median ms | Reduction vs v745 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v745 / c0 | 2.552832 | 2.541824 | 2.536192 | 2.531328 | 2.539008 | 0.0000% |
| v751 / c1, combined | 2.646016 | 2.627072 | 2.624768 | 2.619136 | 2.625920 | -3.4231% |
| v750 / c2, Stage2 only | 2.580736 | 2.575616 | 2.613504 | 2.599424 | 2.590080 | -2.0115% |
| v749 / c3, Stage1 only | 2.593792 | 2.535168 | 2.578688 | 2.562304 | 2.570496 | -1.2402% |

Same-round candidate-minus-v745 deltas, in microseconds:

| Version | Four deltas us | Median delta us | Mean delta us | Faster / tied / slower |
| --- | --- | ---: | ---: | --- |
| v751 | +93.184, +85.248, +88.576, +87.808 | +88.192 | +88.704 | 0 / 0 / 4 |
| v750 | +27.904, +33.792, +77.312, +68.096 | +50.944 | +51.776 | 0 / 0 / 4 |
| v749 | +40.960, -6.656, +42.496, +30.976 | +35.968 | +26.944 | 1 / 0 / 3 |

The first-window benefit does not transfer to this second fixture. Reject v751 as the next OJ candidate and leave v752's unaudited/untested static combination paused; do not hide the second result by pooling windows or extrapolating a score. Prior source files, positive boundary evidence and negative timing samples are retained, with no submission.py promotion.
