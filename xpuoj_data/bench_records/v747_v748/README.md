# v747 / v748: isolate and combine E64 runtime-tail stages

## Status and frozen source identities

**Latest screenshot: OJ140335 Accepted80.33, context-associated with v748;
uploaded source is not verified. This matches, but does not exceed, the
historical highest score. Retain it as a same-score comparison, not a
replacement for all baselines or proof of an isolated E64 gain.** Python/Ruff,
CPU isolation and captured E64 FP32 Stage2 source checks passed. The completed
v748 boundary run below also passed explicit bitwise comparisons. v745,
v746, v747 and v748 have now each completed a full-chain/normal-entry
correctness check and warmed entry timing in both local fixtures. v748 is
faster in7/8 paired rounds against v745, not8/8. No OJ result is
recorded for v747; v748's screenshot feedback is recorded below. Keep
submission.py unchanged. The frozen source
headers' static-candidate statements describe creation time; this README's
later evidence supersedes that status without changing the tested files/hashes.
Local filename/version-reference scans found neither number occupied before
creation. The candidate-authoring subtask did not change parent probes or
Git/GPU/SSH state; subsequent GPU runs were performed by the main thread and
are documented separately below.

| Source | SHA256 |
| --- | --- |
| Final v745 base | `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465` |
| Existing v746, left untouched | `9cd17d1b2b8e02fd59fb277d602e9ad03e654b932aa536b43211075dca7e3416` |
| [v747: Stage2 only](../../probe_v747_v745_e64_stage2_runtime_m64.py) | `c0bb51ebfcd9648329bd884a86fd3958934610e39b5d11f2662707c33643ec6f` |
| [v748: both stages](../../probe_v748_v747_e64_stage1_runtime_m64.py) | `af6b1c88d741d78de3b6a77a00d86afb136c39036c2bd76bea6daa361005ad20` |

## Minimal changes

v747 clones final v745. Only `_get_stage2`'s existing E32 conditional and
`run_kernel`'s two empty-input guards expand to:

```python
num_experts == 32 or (
    num_experts == 64 and hidden == 7168 and intermediate == 2048
)
```

The existing inner Stage2 selection is otherwise unchanged: positive raw/
padded/block counts choose `_moe_stage2_runtime_m64_route_bounds`; empty raw
routes choose `_moe_stage2_e32_zero_output` (the name is historical); a
nonpositive block count retains the clamped full-M128 fallback. No kernel
builder body changes. v747 Stage1 remains exactly v745.

v748 adds only v746's Stage1 selector change, `num_experts == 32` to
`num_experts in (32, 64)`, while retaining the inner exact H7168/I2048 and
positive padded/block conditions. Thus v748 versus v747 isolates that Stage1
dispatch change; it does not duplicate or rewrite either kernel implementation.

| Exact E64/H7168/I2048, normal positive input | Stage1 | Stage2 | Target empty-input extension |
| --- | --- | --- | --- |
| v745 | Original terminal-K GIU M128 | Original dual-B M128 | none |
| v746 (separate prior candidate) | Runtime M128/M64 GIU | Original dual-B M128 | none |
| v747 | Original terminal-K GIU M128 | Existing clamped runtime M128/M64 | yes |
| v748 | Runtime M128/M64 GIU | Existing clamped runtime M128/M64 | yes |

E32 remains exactly v745. E64 neighboring H/I shapes, E16 and other expert
counts retain their original paths and original limits. Normal supported
positive inputs still launch Stage1 then Stage2, computing the current input
each call. Workspace and JIT caches remain allocations/callables, not results.

## Empty-input scope and implementation limits

For the exact E64 target, padded=0 now returns before allocating workspace or
building/launching kernels; raw=0 with padded>0 now makes one no-load zero-
output Stage2 launch and skips Stage1. This mirrors E32's existing handling
and prevents trying to clamp an empty route array. It is deliberately not
applied to all E64 shapes or all expert counts.

The normal positive Stage2 branch uses the existing raw index clamps and
runtime M128/M64/zero-row structure. Full/tail tile sizes, dual-B MMA order,
shared buffers, pass configs, swizzle2, FP32 route multiplication and FP16
output conversion remain the inherited builder's code. The Stage1 runtime
builder likewise retains terminal-K GIU, k_pack2 and E64 swizzle2. Both use
the inherited two shared allocations, not an extra tail shared tile.

This is not a general malformed-input or zero-dimension safety fix. Mock
zero-grid launch requests do not prove device validity. In particular, a
positive route count with an empty block map is not newly validated metadata;
it retains the existing fallback behavior. Unchanged neighboring shapes do
not acquire route clamps or empty-array protection.

v747 changes the chosen Stage2 implementation, including its bounds handling,
branch structure and possible compiler effects, not merely an isolated count
of M64 operations. Any later timing attribution must account for that. Local
E32 evidence does not establish E64 performance or resource allocation. No
async/BSM/pipeline, external device implementation, extra normal launch or
new global workspace is introduced.

## Completed independent CPU checks

Run from the repository root:

```text
python xpuoj_data/bench_records/v747_v748/audit_v747_v748_cpu.py
```

[Audit source](audit_v747_v748_cpu.py) uses only the standard library and:

- Reconstructs both full-module ASTs and complete executable text from v745,
  allowing only the three scoped guards plus v748's one Stage1 selector.
- Verifies every kernel builder and all other functions are text-identical;
  v747's Stage1 selector is v745's and v748's is frozen v746's. Also checks
  the unchanged v746 hash, E64 swizzle2 and zero-builder absence of tensor loads.
- Executes each version's actual host functions with mocked builders/tensors:
  216 shape/dtype/raw/padded/block combinations, two fresh input sets each,
  for v745, v747 and v748. Exact E64 changes from the old two launch requests
  to zero or one in the relevant empty cases; normal positive inputs remain
  two. E32, E64 H/I neighbors and other expert counts remain identical.
- Checks fresh input arguments on each call, output/workspace wiring, stage
  identity, route dtype, and allocation/JIT-only reuse without skipped work.

All assertions, Python syntax and Ruff passed at the SHA values above. These
CPU checks alone are host/source proofs, not generated-code, GPU precision,
bitwise, race-freedom, occupancy or performance results. The completed GPU
boundary evidence below is separate and does not establish an OJ recommendation.

## Independent E64 edge-helper review, before execution results

Reviewed [remote_v746_v748_e64_edges.py](../../remote_v746_v748_e64_edges.py),
SHA256 `e49e9daf064145ecf9973ab71d7e765cded9806080f721256bab389aa952689b`.
No blocking host/dataflow error was found. Pure-CPU extraction of its actual
metadata function confirms SIZES32 repeated twice:64 experts, raw4746,
padded9216,72 CTAs (2 zero-row,36 with1..64 rows,34 with65..128),4470 padding
rows. Valid/padding masks exactly match the expert offset ranges; the final
valid padded row is9088, correctly addressed by output[-128]. This fixture
is not the official OJ distribution or the code-generation fixture below.

The reference is deliberately v745 Stage1 plus a **direct call to v745's
clamped full-M128 Stage2 builder**, not its original unprotected E64 Stage2
dispatcher. Getter and direct-builder arguments, metadata order and masks
match the source signatures. The isolated Stage2 comparison gives both
kernels work_base; only the chain comparison uses work_probe. Stage1 compares
valid rows and requires untouched padding to stay NaN; both Stage2 outputs
require zero padding. There is no timing in this helper.

Gate/Up/Down weights are fixed at seeds74810/74811/74812. Two fresh X/route
sets use74801/74802; route dtypes reuse those underlying values, so this is
not four fully independent all-input sets. `describe` asserts finite output
and tolerance_bad=0, **not bitwise equality**: bitwise is measured with int16
views and separately accumulated into all_tested_bitwise. A final PASS alone
cannot be reported as bitwise success; the actual per-comparison/final flags
must be read. No execution result is inferred from this review.

Empty host checks execute only v748. Getter prohibitors are diagnostic-only
instrumentation and restored in finally; no submission code is instrumented.
v747's run_kernel and Stage2 selection are AST-identical, supporting the
source-level inheritance, but that is not a separate v747 GPU run. The helper
tests target raw-empty and padded-empty behavior, not every neighboring shape.

## Actual E64 FP32 Stage2 code-generation comparison

[v747 E64 Stage2 raw source](codex_e64_747_stage2_codegen_fp32.log) was compared
in full against v743's reviewed FP32 source in the
[E32 combined capture](../v743/codex_e32_723_743_codegen_fp32.log). Both source
hashes were independently recomputed using the recorded source character
counts. After **only** renaming the kernel identifier and replacing the12
scalar route-load clamp bounds4543 with9087, the complete sources are
byte-for-byte identical:

- E32 v743:32771 characters,
  `d318c28eac64a820d1f40d0e10c369592e778e21b1966ff8f258c11c4d535582`.
- E64 v747:32771 characters,
  `abfc6d6d1213a19188d0ca31e726b22bd0cb65c52e2f4f9b5a177f265b454e21`.

Every replaced bound occurs in a scalar raw-route load. The captured E64
signature uses H7168/I2048 and raw9088; normal positive-route bounds differ
from the raw4746 edge fixture above. Shared remains Down0 / Up16384 and six
static barrier sites, three per mutually exclusive positive-row branch.
No other copy, LDS, MMA, synchronization, mask or output-address expression
differs from the prior [v743 source audit](../v743/CODEGEN_AUDIT.md).

This is one FP32-route generated signature, not an E64 FP16-source comparison,
zero-input source audit, numerical test, ISA/resource measurement or timing.
v748's Stage2 builder/selection is identical by the CPU audit; do not label
this v747 capture as a separate v748 compiler run.

## Completed v748 E64 boundary run and raw trace metadata

[Execution log](codex_v748_e64_edges_profile.log), SHA256
`5091b5a90a9d0895b3c99ace8f87cb7e800c5642df0bf5d6a5b8311b7ff9bd34`,
and [trace](edges-1113724.json), SHA256
`95c2fee60c684d9af6857caef7afc98fd570c9e062bb486b8eebbd04edb22812`,
were independently parsed. The remote baseline was v745's tested-header
`12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96`,
not its later comment-only ec864... file; that header-only relationship is
recorded in the v745 archive. The tested v748 SHA matches af6b... above.

On the reviewed raw4746/padded9216 fixture, all10 actual numerical comparisons
report finite=True, max_abs_full=0, nonzero_diff=0, bitwise_equal=True and
tolerance_bad=0: two Stage1 valid-row comparisons (9719808 elements each),
four isolated Stage2 and four composed-chain comparisons (66060288 each).
Both Stage1 padding_untouched_nan checks and all four Stage2 zero_padding
checks pass. The final flag is **all_tested_bitwise=True**; this is read from
the actual results, not inferred from the tolerance PASS. Weight seeds remain
fixed; only the two X/route sets vary, as described in the helper review.

Both route dtypes also pass v748's raw-empty host check with finite all-zero
output and Stage1 getter skipped, and padded-empty checks with workspace
getter skipped. These are real v748 helper calls. The unchanged shared
components support v746 Stage1 and v747 Stage2, but **neither v746 nor v747
was independently run through its complete normal entry** in this experiment.
The reference remains v745 Stage1 plus the explicitly selected clamped M128
Stage2, not the original v745 E64 dispatcher or an independent OJ/math oracle.

Selecting device events with pid2, ph=X and the stage1/stage2_v748_edge names
gives18 events. Each group has consistent args.mem values:

| Role | Events | registers_per_thread | dynamic_shared bytes |
| --- | ---: | ---: | ---: |
| Stage1 reference | 2 | 248 | 32768 |
| Stage1 v748 | 2 | 248 | 32768 |
| Clamped M128 Stage2 reference, both dtypes | 4 | 154 | 32768 |
| Runtime Stage2 isolated/chain, both dtypes | 8 | 154 | 32768 |
| v748 empty-route zero-output Stage2 | 2 | 8 | 0 |

All18 events report static_shared=0, private_per_thread=0 and private_total=0.
Blocks are256x1x1; Stage1 grid72x16x1, Stage2 grid72x56x1. Equal resource
fields and zero private fields do not prove equal occupancy, no spills or
no local-memory traffic. No stall/bandwidth/occupancy counters were collected.
The trace includes cold fixed-order calls, compilation and comparisons; its
durations are not warmed entry samples, and no speedup is derived here.

OJ recommendation remains paused. The main thread reports a user screenshot
for v745 submission140296: Accepted72.67, point1=3733us and rounded point2/3
times7ms/14ms, versus earlier v74380.33 with2567us/4597us/rounded9ms. This
includes a slower unchanged point, so it does not cleanly isolate the code
change. Rounded millisecond displays are not exact microsecond measurements.
A same-window baseline/source-feedback check is requested before promotion;
the successful local E64 boundary run does not override that uncertainty.

## Window1: complete independent E64 random-entry comparison

[Full window1 log](codex_e64_745_746_747_748_random_entry.log), SHA256
`658bdb0aebae544f8f97d991bc48d1259df63e562229e7928e3a9e873f25e594`,
completed through `stage_end stage=entry`; it was not truncated mid-round. Candidate
order is c0=v745, c1=v746, c2=v747, c3=v748. The local random
alternating64-220 fixture has raw9088/padded12288/96 CTAs, different from the
boundary fixture. Reference output is freshly computed by c0's **normal
v745 E64 path**, not the edge helper's explicitly clamped M128 Stage2.

All24 correctness records (4 versions x full-chain/normal-entry x3 same-input
recomputations) report max_abs=0.000000 and bad=0/88080384 after NaN poisoning.
This is tolerance evidence, not entry-level bitwise equality or three random
seeds. All four versions completed one entry warmup before any timing;
16 iters1 samples follow exactly F(0,1,2,3), R(3,2,1,0), F, R. A warmed
entry invocation still contains the normal two GPU kernels; the log's warmup
launch count does not mean a one-kernel implementation. No trace durations
are used here.

| Candidate | Four chronological samples (ms) | Median (ms) | Median latency reduction vs v745 | Paired faster / tied / slower vs v745 |
| --- | --- | ---: | ---: | ---: |
| v745 | 8.900096, 8.913920, 8.893952, 8.940032 | 8.907008 | 0.0000% | baseline |
| v746: Stage1 only | 8.822528, 8.815360, 8.799744, 8.770304 | 8.807552 | 1.1166% | 4 / 0 / 0 |
| v747: Stage2 only | 8.788224, 8.751616, 8.749568, 8.778752 | 8.765184 | 1.5923% | 4 / 0 / 0 |
| v748: combination | 8.624896, 8.670720, 8.648448, 8.587008 | 8.636672 | 3.0351% | 4 / 0 / 0 |

Reduction is 100*(1-candidate_median/base_median), not the reciprocal
speedup. For example, v748's logged speedup+3.130% corresponds to latency
reduction3.0351%. All three candidates are locally positive in this one
window; no stable cross-fixture or OJ benefit is established yet. The main
thread has started a synthetic-routing/random-input second window with
list order745/748/747/746; its unfinished data has not been read or recorded.

OJ recommendation remains paused. A subsequent screenshot reported by the
main thread is submission140309, Accepted80, point1=2568us, point2=4599us,
point3 displayed as rounded9ms. Its assignment to v743 follows conversation
context and is **not source-verified**. That suggests baseline-like timing
has returned, but neither identifies why the earlier v745 window regressed
nor verifies the candidate source. The same v745 file has been requested for
retest. These local gains do not replace that feedback check.

## Window2 completed; v748 selected for manual OJ validation

[Full window2 log](codex_e64_745_748_747_746_synthetic_random_entry.log), SHA256
`d49531acda3ffe4ec65c0a83450728bccaa639187ad6eb24a1add819b7666647`,
completed all three correctness rounds and16 timing samples through
`stage_end stage=entry`. This second fixture uses synthetic routing with
**random tensor values**, padded11136, not constant inputs. The list is now
c0=v745, c1=v748, c2=v747, c3=v746. Its24 full-chain/entry comparisons all
report max_abs=0.000000, bad=0/79822848 against saved
`/root/ref_v432_case3.pt`, including the same-window v745 control. It is not
a fresh-c0 reference or an independent OJ oracle. The three NaN-poisoned
recomputations reuse one input batch; entry bitwise equality was not checked.

All four warmup1 phases precede four F/R/F/R rounds, iters1. Samples are ms;
reductions use1-candidate_median/base_median, not reciprocal speedup.

| Candidate | Four chronological samples (ms) | Median (ms) | Latency reduction vs v745 | Paired faster / tied / slower |
| --- | --- | ---: | ---: | ---: |
| v745 | 8.216064, 8.228864, 8.171776, 8.259584 | 8.222464 | 0.0000% | baseline |
| v748: combination | 8.174848, 8.204032, 8.185344, 8.161536 | 8.180096 | 0.5153% | 3 / 0 / 1 |
| v747: Stage2 only | 8.191232, 8.377344, 8.152832, 8.153600 | 8.172416 | 0.6087% | 3 / 0 / 1 |
| v746: Stage1 only | 8.157696, 8.256000, 8.171520, 8.203008 | 8.187264 | 0.4281% | 3 / 0 / 1 |

v748 loses round3 versus v745 by13.568us; it is not the fastest separate
median in window2. v747's8.377344ms sample is retained. The combination's
two-fixture median reductions are3.0351% and0.5153%, with4/4 then3/4 wins
(7/8 total). This supports a bounded OJ experiment, not a guaranteed gain or
stable additive contribution from the two stages. No trace samples are pooled
into these entry measurements. Both runs ended and the main thread confirmed
the GPU lock released.

The main thread now reports the requested v745 repeat screenshot140316:
Accepted80, point1=2565us, point2=4530us, point3 rounded9ms. Relative to the
context-associated v743140309 point2=4599us, that is69us (about1.50%) lower;
the severe72.67 window did not recur. These screenshot-to-version mappings
remain context-based, **not source-verified**, and rounded9ms is not an exact
microsecond result. The prior pause is therefore replaced by **v748 pending
user manual OJ submission**, while retaining the feedback/identity caveats.
v748 inherits v745's E32 implementation unchanged; only the specified E64
paths differ. That pending recommendation is superseded by the feedback below.

## OJ140335 screenshot feedback (2026-09-05):80.33, matching the prior high

The user's screenshot reply to the v748 code request shows
[submission140335](https://xpuoj.com/contest/5/submissions/140335), Accepted80.33,
total16ms and memory22.3G. The v748 association is inferred from reply
context, not explicit version text or uploaded-source verification. The
[structured transcription](oj_140335_user_report.json) records both
`version_mapping_explicit=false` and `uploaded_source_verified=false`.
The [screenshot](oj_140335_user_screenshot.png) SHA256 is
`71824c9ffc15fb5d4b5156cbdd71e5f324886103daf228a7418d421a32654c1a`.

| Item | Status | Displayed time | Exact microseconds | Point score |
| --- | --- | --- | ---: | --- |
| Sample1 (not scored) | Accepted | 2539us | 2539 | Not applicable |
| Formal1.1 | Accepted | 2551us | 2551 | Unknown |
| Formal1.2 | Accepted | 4489us | 4489 | Unknown |
| Formal1.3 | Accepted | 9ms | Unknown | Unknown |

Compared with context-associated v745/140316 (80,2565us/4530us/rounded9ms),
formal point1 is14us lower and point2 is41us lower. Those E16/E32 paths
are unchanged by v748, so these differences are not attributed to E64.
Point3's exact microseconds and all individual point scores are unknown;
the rounded9ms display cannot independently establish the new E64 path's
benefit or an exact total-time difference.

The overall score rises0.33 relative to that v745 repeat and matches the
historical high80.33; it does not set a new high. Retain v748 as a same-score
comparison while preserving prior baselines and source-identity caveats.
No API query, login, browser action or code submission was performed for
this screenshot transcription.

## OJ140368 screenshot feedback (2026-09-05): v747 association, Accepted80

The user's reply to the v747 recommendation shows Accepted80, total16ms,
memory22.3G. Sample2633us is separate from formal2633us/4614us/display9ms.
The association is contextual, not explicit version text or uploaded-source
verification. [Structured transcription](oj_140368_user_report.json) and
[original screenshot](oj_140368_user_screenshot.png) retain these limits;
screenshot SHA256 is
`f8c964039081bc7792adafac86860c07ed6a4d51da8a64212e49abe954833706`.

Full local source comparison confirms v747/v748 differ only in the E64
Stage1 selector; all other functions, imports and top-level caches match.
Relative to v748/140335, unchanged E16/E32 are82us/125us slower (3.2144%/
2.7846%). Neither the cause of this variation nor the effect of the E64
change is isolated. Two rounded9ms displays cannot rank E64 performance,
and total-score80 versus80.33 does not identify any individual point score.

Do not promote v747 as faster. Retain prior80.33 baselines, including v748.
Request existing submissions' detailed point3 feedback if the UI exposes it,
not another duplicate submission. No API/login/browser/submission action was
performed. The E16 v749-v751 candidates were separately rejected after the
second local routing fixture; v752 remains an untested, stopped static draft.
