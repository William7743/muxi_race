# v744: all-M128 static-K / direct-row-guard control

2026-09-05. Diagnostic control on v723; no OJ submission or score.
`submission.py` and all earlier probes remain unchanged.

## Design and identity

Tested source: `../../probe_v744_v723_e32_stage2_static_k_guard.py`.
SHA256: `e6345cc2dd0ca5c39aeda3712757a3ea18de6cf1bf08ac1f61c6037cdbe2736a`.

Final archived source SHA256 after the result-header update:
`4cf9d688af3089b7d01584ac4857753c747a9481f5d5085648a463366b50c310`.
Only one unverified-status comment was replaced by two result comments.
Both the main thread and this audit independently verified that reversing
those comments reproduces the exact tested SHA above and preserves the full
AST. The final byte SHA was checked locally. The tested SHA is not replaced
by the final-header identity; no changed executable logic was retested.

Only three executable changes in `_moe_stage2_fast_bfrag_prefetch_route_bounds`:

1. Remove `active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(intermediate, be2), 0)`.
2. Replace its `if active_k_steps > 0` guard with `if actual_rows > 0`.
3. Replace the steady loop bound with `range(T.ceildiv(intermediate, be2) - 1)`.

Every positive block still computes M128, including short tails. No M64
branch, new buffer, altered MMA/epilogue, extra launch or changed shared layout
is introduced. Pass configs, 256 threads, 2x2 warp64x64 geometry, synchronous
copy/clear order, dual-B fragment lifetimes, raw-route clamps, Stage1, caches
and host dispatch remain v723. The changed builder is the existing positive-
route E32 selection, including E32 neighboring dimensions that already select
it; there is no new H7168/I2048-only dispatcher. Non-E32 code is unchanged.

This tests whether the compiler simplifications observed in v743's full branch
can help without reducing M work. It is not a perfectly isolated M64 effect:
v743 additionally swaps the physical Up/Down shared bases, branches on rows>64,
changes the full epilogue's inferred range and adds an M64 branch/private
fragments. Timing comparisons cannot assign all remaining difference to one
of those factors without further evidence.

## Historical de-duplication

`probe_v553_s2_e32_static_k_diagnostic.py` differs from v545's dual-B builder
only by adding Python `k_steps=(intermediate+be2-1)//be2` and changing the loop
to `range(k_steps-1)`. It **retains** the `active_k_steps` assignment and its
outer conditional; v744 removes that variable and uses a direct row guard.
v553 also predates the inherited v723 route-address clamps and empty-route
dispatch. It is a close static-bound control, not this exact transformation.

The historical OPTIMIZATION_LOG records Stage2 median 1.867136 ms versus
v545 1.849259 ms, about 0.97% slower. No matching original v553 timing log was
found locally during this review; do not reinterpret its historical
"elementwise identical" wording as an independently verified bitwise result.
This prior negative result motivates a small causal diagnostic, not promotion
of the same static-K idea without fresh evidence.

## Independent CPU audit

[Executable audit](audit_v744_cpu.py) passed, as did Python syntax and Ruff.
It uses standard-library code only and runs no device kernels.

- Full-module AST and executable text match v723 after exactly the three edits
  above and the leading comment header; every other function remains unchanged.
- For rows0..128 and K64-tile counts1,2,32, the actual compute AST produces the
  same ordered copy/clear/LDS/MMA/explicit-WAR-barrier tags in v723 and v744.
  Positive rows accumulate K16 tiles in unchanged order; zero rows load no
  inputs and read no accumulator in this Python-level trace.
- Execute the unchanged actual epilogue over all row counts, first/later raw
  groups and final/nonfinal groups: every output is written once; valid rows
  select their original raw row; invalid rows are zero. Positive-length clamps
  stay in range, including potentially hoisted invalid-row addresses.
- Reuse only `Tensor` / `host_mock` helper definitions from the v743 audit,
  testing each version's own unchanged host AST: 168 shape/empty-route/
  empty-padded/empty-block-map/dtype combinations, two fresh calls per case.
  Dispatch, arguments, 0/1/2 launch requests and workspace/JIT-only reuse match.

**Shape limits:** an E32 zero-padded input returns through the unchanged host
path; zero total route length selects the old no-load zero kernel. A positive-
route zero-row CTA is different from an empty route array. For positive rows
with `intermediate=0`, the original active-K guard is false but the new row
guard is true; the audit explicitly detects this non-equivalence. Such zero-K
input is not added to the supported baseline contract. Zero-grid host mocks
also do not prove a GPU accepts zero-sized launches or malformed metadata.

## Actual FP32 code-generation comparison

Compare [v744 raw codegen](codex_e32_744_codegen_fp32.log) against the v723
section of [the prior combined log](../v743/codex_e32_723_743_codegen_fp32.log).
Both captures use E32/H7168/I2048, raw4544/padded6144, 48 M blocks and FP32 route
weights. Source extraction uses each metadata record's `source_characters`;
the independently recomputed source hashes match the records.

| Capture | Characters | Generated source SHA256 |
| --- | ---: | --- |
| v723 | 18278 | `36c433f89d966c69fddcc9d6a015f954c6cb8a7ce5a045c78b8d31a687af5aa7` |
| v744 | 17188 | `350cc833690194783548c397b70d3c4058eb6e6bbc62ed92cfd0ae5a1c66edb7` |

The complete generated diff shows:

- `condval`, `condval_1` and `condval_4` assignments disappear; the two compute
  regions use the direct block-uniform positive-remaining-row guard.
- Steady K becomes literal `k<31`. Both next-copy `if(k<31)` / zero temporary
  branches disappear. Up-next indexing uses 32-bit instead of 64-bit integer
  casts; the reviewed Up allocation's maximum half-element index is12582911,
  within signed32 range. The archived integer index expressions have identical
  AST algebra after removing only integer casts. Down-next indexing retains
  its original wider casts. Shared destinations match exactly.
- All16 actual LDS assignment lines and all8 complete static MMA statements
  are byte-identical after trimming indentation. Shared bases remain Up0 /
  Down16384, unlike v743's swap; shared/private declarations match v723.
- Both route-load statements are unchanged and individually clamp raw indices
  to0..4543. The entire epilogue matches except one zero-temporary variable
  name; output ownership, masking, FP32 multiply and FP16 conversion are intact.
- Three static barrier sites remain. Source-relative lines42/119/130 are loop-
  head RAW, post-MMA/pre-overwrite WAR, and terminal RAW; the v723 counterparts
  are54/131/156. The terminal barrier stays **outside** the positive-row guard,
  so an empty-row CTA still reaches this one generated barrier, unlike v743's
  explicit zero branch. Positive blocks execute63 source-level barrier calls
  for K32, not three; this is not a machine-level counter measurement.

The optional codegen section in the CPU audit rechecks the archived hashes,
shared/private declarations, guards, loop bound, all LDS/MMA statements, next-
copy address algebra, route loads and normalized epilogue. No definite source
dependency error was found. Reported register/spill fields are null, not zero;
32 KiB shared and local array sizes do not prove physical register usage or
occupancy. This capture alone establishes neither FP16 generated identity nor
GPU numerical correctness, timing or OJ acceptance.

## Completed Stage2 edge checks

[Raw edge log](codex_v744_stage2_edges.log) uses the v743 edge helper with v744
as candidate and v723 as baseline. E32/H7168/I2048, raw2373/padded4608, 36 CTAs,
including rows0/1/63/64/65/127/128 and the final raw token. Up padding and output
are NaN-poisoned. Both route dtypes (FP32/FP16), each with Up/route seeds74301
and74302, pass finite=True, max_abs_full=0, nonzero_diff=0,
bitwise_equal=True, padding_nonzero=0. The helper explicitly compares output
int16 views; this result is genuinely bitwise for this Stage2 fixture.

Down weights remain fixed at seed74300; switching route dtype reuses the same
two underlying Up/route seeds. This is not four fully independent all-tensor
input sets, an independent mathematical oracle or an OJ correctness result.
The fixture includes an allocated zero-row CTA, not zero-total-route or
zero-padded GPU input. Those host paths are inherited and separately bounded
by the CPU audit above.

## Completed untraced random-entry cohort

[Complete entry log](codex_e32_720_723_744_743_random_entry.log) compares
c0=v720, c1=v723, c2=v744 and c3=v743 on the local random
`alternating64-220` fixture: raw4544/padded6144/48 CTAs. This is one window,
not a multi-seed confirmation. All four versions passed three same-input
NaN-poisoned recomputations of both full chain and actual `run_kernel`, relative
to fresh c0 output: `max_abs=0.000000`, `bad=0/44040192`. These are tolerance
checks with a rounded max printout, **not** entry-level bitwise equality; the
explicit bitwise result above applies only to the independent Stage2 fixture.

Timings use warmup1, iters1 and four forward/reverse/forward/reverse rounds.
All samples below are milliseconds; diagnostic edge/compile/trace time is
excluded. No A/A duplicate is present in this cohort.

| Candidate | Round 1 (F) | Round 2 (R) | Round 3 (F) | Round 4 (R) | Median (ms) | Mean (ms) | Stddev (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 4.663808 | 4.689152 | 4.657920 | 4.686848 | 4.675328 | 4.674432 | 0.013751 |
| v723 / c1 | 4.664576 | 4.675840 | 4.602624 | 4.631808 | 4.648192 | 4.643712 | 0.028712 |
| v744 / c2 | 4.671744 | 4.684800 | 4.712704 | 4.662272 | 4.678272 | 4.682880 | 0.018986 |
| v743 / c3 | 4.555264 | 4.584448 | 4.601856 | 4.551424 | 4.569856 | 4.573248 | 0.020878 |

Paired deltas are candidate minus control in microseconds; negative means
faster. The median of paired differences is not a difference of medians.

| Pair | Per-round delta (us) | Median delta (us) | Mean delta (us) | Faster / tied / slower |
| --- | --- | ---: | ---: | ---: |
| v744 - v720 | +7.936, -4.352, +54.784, -24.576 | +1.792 | +8.448 | 2 / 0 / 2 |
| v744 - v723 | +7.168, +8.960, +110.080, +30.464 | +19.712 | +39.168 | 0 / 0 / 4 |
| v743 - v720 | -108.544, -104.704, -56.064, -135.424 | -106.624 | -101.184 | 4 / 0 / 0 |
| v743 - v723 | -109.312, -91.392, -0.768, -80.384 | -85.888 | -70.464 | 4 / 0 / 0 |
| v743 - v744 | -116.480, -100.352, -110.848, -110.848 | -110.848 | -109.632 | 4 / 0 / 0 |
| v723 - v720 | +0.768, -13.312, -55.296, -55.040 | -34.176 | -30.720 | 3 / 0 / 1 |

v744's median is 0.0630% slower than v720, with mixed pairs, and 0.6471%
slower than v723. v743 is 2.2559% faster by median elapsed time versus v720
and 2.3174% versus v744 in this window. Its logged `speedup_vs_c0=+2.308%`
is the reciprocal speed ratio, not a 2.308% reduction in time.

## Decision

**Do not recommend v744 for OJ; stop after this fixture.** No v744 second-
fixture test, OJ submission ID, Accepted result or score is claimed. The
static-K/direct-row-guard control does not reproduce v743's local entry gain
in this window. This weakens a claim that those compiler simplifications
alone explain v743, but does not prove all remaining gain comes solely from
M64 work reduction: shared-base placement, branch assumptions/code structure,
resource allocation and short-window measurement variation remain factors.
The third-round v723 sample is notably faster than its other samples; all
raw values are retained and no sample was dropped.

Keep the existing OJ baseline and `submission.py` unchanged. The **tested**
and comment-only **final** identities are both recorded above; raw edge,
entry and code-generation logs are archived in this folder. v744's local
test sequence is closed and no further v744 GPU run is planned.
GPU scheduling and subsequent experiments remain the main thread's job;
this folder's audit does not claim to acquire or release the shared GPU lock.
