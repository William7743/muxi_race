# v743: one Stage2 kernel with runtime M128 / M64 / zero paths

2026-09-05 experiment. User identifies OJ submission140270 as v743; the supplied screenshot shows Accepted,80.33. Uploaded-source identity and precise point3 time are not verified. Keep `submission.py` unchanged.

## Hypothesis and design

The E32 Stage2 row blocks include short tails. A CTA-uniform branch can halve
the M dimension for 1–64 valid rows without adding a launch or changing the
external M128 block map. This may reduce tail A traffic and MMA work; it does
not reduce Down-weight traffic or guarantee lower resource allocation.

Base is **v723**, not an unprotected v720 clone: retain its clamped E32 raw
route-weight addresses and its zero-valid/zero-padded host paths. Only positive
E32/H7168/I2048 inputs with nonempty padded rows and block map select the new
builder. All old builders and all other host behavior remain unchanged.

Inside one `T.Kernel`, rows>64 use the original v723 M128 dual-B calculation;
1–64 rows use the v634 M64 dual-B emitter sequence with an independent C layout;
zero rows only write all 128 output rows to zero. Both paths share one
128x64 Up tile and one 128x64 Down tile (32 KiB). The M64 path uses the first
64 Up rows and explicitly zeros output rows 64–127. It uses 2x2 warps,
warp32x64, k_pack1; the full path retains warp64x64. Threads remain 256.

This differs from v574/v581/v634's separate main/tail callables and extra
launches. It is not a claim that those older local gains reproduced on OJ.
Positive inputs still launch Stage1 then Stage2, recomputing current inputs.
No async/BSM/pipeline DSL, external device computation or historical result
replay is introduced. Torch in the independent test helper is diagnostic only.

## Frozen identity and verification scope

Tested probe SHA256:
`67d25409b20cf6417d79375f57edb3770a79fb1a7a619eb8bc3ca9e3b6e0e7ec`.

Final delivery probe SHA256 after the completed tests:
`5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9`.
Only two introductory status-comment lines were replaced by three result
comment lines. The main thread verified that reversing this comment update
reproduces the tested SHA exactly and that the complete AST is unchanged.
The final SHA was also independently checked locally. The tested identity
above is retained; the final header is not a separately GPU-tested program.

Remote v723 control SHA256:
`4f749f45ea547ce36a16f143e5667bf7d3c3f27a6efeabf7a3ec1c30f5ac9235`.
It is the original GPU-tested v723; the repository's
`c780850f58c3b29b4663ca8a5aafd5285b05e7d8f8abe4e10b9ce34bf2ca3fe1`
differs only in the result header, as recorded in the v723 evidence.

Diagnostic helper `../../remote_v743_stage2_edges.py` SHA256:
`eff7a97973304bb02639ebe0e652e0088bcfcdb0044621734c291281dc8d55c1`.

- [Independent CPU audit](audit_v743_cpu.py): source/AST inheritance, host
  dispatch, row/K tags, layout and output-address coverage. Not a GPU proof.
- [FP32 code generation](codex_e32_723_743_codegen_fp32.log) and
  [FP16 code generation](codex_e32_743_codegen_fp16.log): compilation passed.
  Captured signature uses raw4544/padded6144/48 CTAs. Both variants have six
  static barrier sites, three in each mutually exclusive positive-row path;
  arrays and static sites are not physical register or dynamic barrier counts.
- Dedicated Stage2 edge fixture uses actual H7168/I2048/E32, raw2373/padded4608,
  36 CTAs: 1 zero, 18 M64, 17 M128. Includes row counts 0,1,63,64,65,127,128
  and a final raw token followed by 127 padded rows. Up padding is NaN-poisoned.
  Two independent random Up/route input sets (seeds74301/74302) are checked
  for each FP32/FP16 route dtype. Down weights use fixed seed74300. The helper
  requires finite output, explicit int16-view bitwise equality to fresh v723
  output, zero numerical differences and zero padding. This is Stage2-only
  baseline comparison, not independent mathematical or official OJ validation.

## Completed independent Stage2 edge checks

[Raw edge/profile log](codex_v743_stage2_edges_profile.log): both independent
Up/route sets for both route dtypes passed with finite=True, max_abs_full=0,
nonzero_diff=0, bitwise_equal=True and padding_nonzero=0. These explicit bitwise
checks do not rely on a rounded maximum-difference printout. This fixture has a
zero-row CTA but does not GPU-test total_valid_tokens=0 or padded=0; those host
paths are unchanged and covered by the 168-case CPU mock audit (two fresh calls
per case). The prior v723 GPU empty-input checks remain separate evidence.

The tracer finished normally at 10:57:45 UTC. It produced
`codex_profile_743_edges_20260905/edges-1106109.json`; this capture includes
diagnostic comparisons and cold first launches, not steady benchmark samples.

## First untraced random-entry cohort: preliminary positive signal

[Raw first-cohort log](codex_e32_720_720_723_743_random_entry.log) compares
candidate0=v720 A, candidate1=v720 B (the same source loaded separately),
candidate2=v723 and candidate3=v743. This is the full `run_kernel` entry, not
the traced Stage2-only diagnostic above. The local `alternating64-220` fixture
has padded6144/48 CTAs; it is not official OJ input. Warmup1, iters1, four
rounds in forward/reverse/forward/reverse candidate order; no profiler is
included in these times.

All four candidates passed three recomputations of the **same random input
batch**, after NaN-poisoning workspace/output, for both `launch_full` and
`run_kernel`. Each comparison reports `max_abs=0.000000` and
`bad=0/44040192`. These are the harness's tolerance checks, not a bitwise test
or three independent random input sets. The separate edge test above supplies
explicit bitwise evidence only for its own Stage2 fixture.

All times below are milliseconds; samples retain chronological round order.
The standard deviations are those reported by the harness.

| Candidate | Round 1 (F) | Round 2 (R) | Round 3 (F) | Round 4 (R) | Median | Mean | Stddev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 A / c0 | 4.646144 | 4.649016 | 4.649984 | 4.654592 | 4.649500 | 4.649934 | 0.003038 |
| v720 B / c1 | 4.683264 | 4.694784 | 4.637952 | 4.649728 | 4.666496 | 4.666432 | 0.023331 |
| v723 / c2 | 4.678656 | 4.646912 | 4.673536 | 4.645888 | 4.660224 | 4.661248 | 0.014962 |
| v743 / c3 | 4.600320 | 4.572160 | 4.544768 | 4.582400 | 4.577280 | 4.574912 | 0.020111 |

Paired deltas are **left candidate minus right control**, in microseconds;
negative means faster. Median-of-deltas is not difference-of-medians.

| Pair | Per-round delta (us) | Median delta (us) | Mean delta (us) | Faster / tied / slower |
| --- | --- | ---: | ---: | ---: |
| v743 - v720 A | -45.824, -76.856, -105.216, -72.192 | -74.524 | -75.022 | 4 / 0 / 0 |
| v743 - v720 B | -82.944, -122.624, -93.184, -67.328 | -88.064 | -91.520 | 4 / 0 / 0 |
| v743 - v723 | -78.336, -74.752, -128.768, -63.488 | -76.544 | -86.336 | 4 / 0 / 0 |
| v720 B - v720 A (A/A) | +37.120, +45.768, -12.032, -4.864 | +16.128 | +16.498 | 2 / 0 / 2 |
| v723 - v720 A | +32.512, -2.104, +23.552, -8.704 | +10.724 | +11.314 | 2 / 0 / 2 |

By the ratio of the separate medians, v743's entry time is lower by 1.5533%
versus v720 A, 1.9118% versus v720 B and 1.7798% versus v723. The first
comparison's logged `speedup_vs_c0=+1.578%` uses the reciprocal ratio and is
not the same percentage as elapsed-time reduction. The identical-source A/A
controls differ by 0.3655% in median time and switch relative order across
rounds, so this cohort does not establish a general v723 regression.

The consistent v743 direction against all three controls is a preliminary
positive signal, not a confirmed OJ improvement. The next section records
the repeat window separately; do not combine this table with traced/cold
edge launches or infer gains in non-E32 paths that were not changed.

## Repeat random-entry cohort: same fixture, changed candidate position

[Raw repeat-cohort log](codex_e32_720_743_723_repeat_random_entry.log) uses
candidate0=v720, candidate1=v743 and candidate2=v723, with no second v720 A/A
copy. The candidate is now between the controls rather than last. This is
a separate untraced timing window with the **same seeded random fixture**
and `alternating64-220` geometry, not a second independent random dataset.
All three candidates passed three same-input NaN-poisoned recomputations
of both full chain and real entry relative to fresh c0 output, each reporting
`max_abs=0.000000`, `bad=0/44040192`; again this is tolerance evidence, not
an explicit bitwise comparison. Warmup1, iters1, four F/R/F/R rounds.

| Candidate | Round 1 (F) | Round 2 (R) | Round 3 (F) | Round 4 (R) | Median (ms) | Mean (ms) | Stddev (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 4.694272 | 4.653568 | 4.659712 | 4.662016 | 4.660864 | 4.667392 | 0.015823 |
| v743 / c1 | 4.534528 | 4.581120 | 4.565504 | 4.540928 | 4.553216 | 4.555520 | 0.018766 |
| v723 / c2 | 4.652800 | 4.675328 | 4.641792 | 4.669440 | 4.661120 | 4.659840 | 0.013298 |

Samples are milliseconds. Paired deltas below use the same candidate-minus-
control convention as the first cohort, in microseconds.

| Pair | Per-round delta (us) | Median delta (us) | Mean delta (us) | Faster / tied / slower |
| --- | --- | ---: | ---: | ---: |
| v743 - v720 | -159.744, -72.448, -94.208, -121.088 | -107.648 | -111.872 | 4 / 0 / 0 |
| v743 - v723 | -118.272, -94.208, -76.288, -128.512 | -106.240 | -104.320 | 4 / 0 / 0 |
| v723 - v720 | -41.472, +21.760, -17.920, +7.424 | -5.248 | -7.552 | 2 / 0 / 2 |

Ratios of the separate medians give 2.3096% less entry time versus v720 and
2.3150% versus v723. The logged reciprocal speedup versus c0 is +2.364%,
not a 2.364% reduction in time. v743 is faster in all four same-round pairs
against either control in this window; the repeated direction survives its
changed list position. These short cohorts still sample only one random
fixture and are not an OJ result or proof across unseen routing distributions.

## Second fixture / third cohort: synthetic routing, random tensor values

[Raw synthetic-routing random-input log](codex_e32_720_743_723_synthetic_random_entry.log)
retains the repeat window's candidate order: c0=v720, c1=v743, c2=v723.
Here `synthetic` names the routing distribution, **not constant inputs**:
the log's CPU random allocation and the harness's saved-reference branch
correspond to `routing=synthetic, input_mode=random`. The case2 geometry,
derived from `remote_bench.make_group_sizes`, is ten experts with 112 rows,
fourteen with 156 and eight with 155: raw4544/padded6912/54 CTAs, comprising
22 full128 blocks, ten 112-row blocks and 22 tails with 27 or 28 rows.

Unlike the fresh-c0 references in the first two cohorts, this harness compares
with the saved `/root/ref_v432_case2.pt`. **All three candidates, including
v720**, passed three same-input NaN-poisoned recomputations of both full chain
and real entry against that reference before timing: `max_abs=0.000000`,
`bad=0/49545216`. This does not authenticate the saved reference as an official
mathematical oracle and does not establish bitwise equality. The three rounds
reuse one generated random input batch; they are not three independent seeds.

Untraced entry timings use warmup1, iters1, four F/R/F/R rounds. Every sample
below is in milliseconds; no trace/cold edge sample is mixed into the table.

| Candidate | Round 1 (F) | Round 2 (R) | Round 3 (F) | Round 4 (R) | Median (ms) | Mean (ms) | Stddev (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 / c0 | 5.193728 | 5.202432 | 5.223424 | 5.218816 | 5.210624 | 5.209600 | 0.012035 |
| v743 / c1 | 5.146112 | 5.168896 | 5.173504 | 5.147136 | 5.158016 | 5.158912 | 0.012401 |
| v723 / c2 | 5.177344 | 5.209856 | 5.231360 | 5.197824 | 5.203840 | 5.204096 | 0.019567 |

Paired candidate-minus-control deltas are microseconds; negative is faster.

| Pair | Per-round delta (us) | Median delta (us) | Mean delta (us) | Faster / tied / slower |
| --- | --- | ---: | ---: | ---: |
| v743 - v720 | -47.616, -33.536, -49.920, -71.680 | -48.768 | -50.688 | 4 / 0 / 0 |
| v743 - v723 | -31.232, -40.960, -57.856, -50.688 | -45.824 | -45.184 | 4 / 0 / 0 |
| v723 - v720 | -16.384, +7.424, +7.936, -20.992 | -4.480 | -5.504 | 2 / 0 / 2 |

v743's median entry time is lower by 1.0096% versus v720 and 0.8806% versus
v723. The logged +1.020% `speedup_vs_c0` is the reciprocal speed ratio, not
the time-reduction percentage. Both comparisons have four faster paired
rounds; the v723/v720 control comparison remains mixed.

## Decision and limits

**Recommend v743 as the next candidate for the user's manual OJ submission.**
Across two local routing fixtures and three timing windows, it is faster in
12/12 paired rounds versus each window's primary v720 control, with median
entry-time reductions of 1.5533%, 2.3096% and 1.0096%. The first window's extra
v720 A/A copy is a useful control, not an additional independent cohort.
The v723 comparisons also support the runtime-tail change rather than treating
the inherited route-index clamp alone as the explanation.

This is still short local evidence: warmup1/iters1, only two routing fixtures,
reused seeds within the repeated fixture, shared hardware state and some A/A
variation. It does not guarantee OJ improvement, unseen-input correctness or
performance in E16/E64 paths, which were not changed. Explicit bitwise evidence
is confined to the separate Stage2 edge fixture; entry checks above use the
harness tolerance. Profiling evidence stays separate in [PROFILING.md](PROFILING.md).

The user subsequently reported **v743 -> OJ submission140270**, and supplied a
screenshot showing **Accepted,80.33**. This is screenshot evidence, not an
authenticated API result; the uploaded-source hash remains unverified. Keep
`submission.py` and the existing OJ baselines unchanged. The user submits the
standalone probe manually; subsequent OJ feedback must be tied to its actual
source identity before any promotion. All local tests have finished, their
raw logs are archived here, no GPU job remains and the GPU lock has been
confirmed released by the main thread.

## User-reported OJ mapping (2026-09-05)

- Submission: [140270](https://xpuoj.com/contest/5/submissions/140270).
- Reported version: v743. The delivered repository artifact is the final
  source SHA5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9
  at commit ec774f8d5390f60cf9764b8f03eb01d731631191.
- The actual uploaded source has not been retrieved/hashed. The subsequent
  screenshot supplies the result below; local tests are not used to fill OJ fields.

## Screenshot-confirmed OJ result

[User screenshot](oj_140270_user_screenshot.png) and [structured transcription](oj_140270_user_report.json):
Accepted,80.33 overall; formal points1.1/1.2/1.3 all Accepted with displayed
times **2567us / 4597us / 9ms**. Point3 is only displayed in milliseconds;
its exact microseconds and per-point scores remain unknown. Sample1 is
separately Accepted at2569us and is not counted as a fourth scored point.
The page displays total16ms and memory22.2G; preserve the display precision.

Compared with v720/139770's recorded point2 of4594us, v743 is3us slower in this
submission, essentially tied rather than reproducing the local gain. The total
score remains80.33. Point1 differs by+18us despite its unchanged code, illustrating
cross-run variation; point3 cannot be compared precisely from this screenshot.
Do not promote v743 as an OJ speed improvement or infer the v745 result from it.

## Requested baseline repeat: OJ140309

After v745/OJ140296 returned72.67, the user replied to the v743-repeat request
with [this screenshot](oj_140309_user_screenshot.png):140309,Accepted,80;
formal points2568us/4599us/rounded9ms, sample2573us, total16ms/memory22.2G.
[Structured record](oj_140309_user_report.json) marks the v743 association as
context-inferred rather than an explicit version message or source check.
The first two points are only1us/2us above140270, so the baseline is back in
its prior timing range. Exact point3 timing and individual scores are unknown;
do not assign the0.33 score difference to a particular point without evidence.
Next diagnostic is a repeat of the identical v745 artifact, not a new probe.
