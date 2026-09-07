# NSA C500 64 GB baseline setup — 2026-09-07

MOE v748 and its final delivery ZIP remain unchanged. No NSA OJ submission made.

## Latest update: exact reference submission

NSA080 online128/shared probability syntax PASS but representative compile
fails GEMM layout Divide by zero; no timing. See ONLINE080.md. Keep078.

NSA079 paired0799/9 PASS, changed S8 cases about10% slower than078;
not promoted. See UNROLL079.md. NSA078 retained.

Current local working candidate: NSA078. Public109 PASS, direct057
sparse repeat confirms changed-path gain, expanded edges108/108 PASS,
max_abs0.00390625. NSA057 preserved. public078repeat seed432 exact109/109
PASS, geomean1.075434492x, cumulative1.103141546x vs frozen reference.
This is local evidence, not OJ88; see ONLINE078.md.

NSA078 public078 exact109/109 PASS, geomean1.074751073x vs frozen;
direct057 sparse repeat confirms1.246x/1.274x changed S8 gains.
edges078108/108 PASS; NSA057 preserved as previous candidate.
See ONLINE078.md; promising local result, no OJ score claim.

NSA077 sparse0779/9 PASS; changed large S8 cases essentially parity,
not promoted. Source077 S8 representative export byte-identical to057;
stage-count change had no code effect there. See ONLINE077.md.

Targeted profile057d64 completed, workload PASS/source hash confirmed.
Counters retained but coverage/units do not establish a bottleneck; see
PROFILE057D64.md. Achieved waves4020 is not an occupancy percentage.

NSA076 early V fragment source verified (V load precedes QK), syntax PASS;
d64076 seed427 exact29/29 PASS, geomean0.786400613x, cumulative0.689504406x;
rejected. See EARLY076.md. NSA057 retained.

NSA075 V fragment source export succeeds, barriers3 vs5 on D64 G16.
d64075 seed426 exact29/29 PASS, geomean0.808795302x, cumulative0.740561899x;
rejected. V scalar source loads replace uint4. See VFRAGMENT075.md.

NSA074 isolated D64/BS16 storage-rewrite-off JIT: syntax PASS,
d64074 seed425 exact29/29 PASS, geomean0.990849370x, cumulative0.996748124x;
no gain, not promoted. Synchronization enabled. See STORAGE074.md.
Follow-up074 D64 B4/L1024 source byte-identical to057: this flag did not
change generated allocation or barriers for that configuration. Hypothesis
not ruled out; compiler-flag candidates should get source comparison first.

NSA073 D64/BS16 CK1 CV2 d64073 seed424 exact29/29 PASS;
geomean0.969407459x, cumulative0.975027299x, not promoted. See PVSPLIT073.md.

NSA068 post-QK mask: d64068 exact29/29 PASS, geomean0.979076744x vs
reference; no benefit, not promoted. NSA069 direct FP16 normalized
probability write (parent057): d64069 exact29/29 PASS, geomean0.996570865x,
no gain. NSA070 transposed shared V29/29 PASS but geomean0.810117694x,
rejected. source057simple inspection completed: shared workspace aliasing
and five barriers. NSA071 adjacent-lane softmax layout syntax PASS,
d64071 seed422 terminal: compilation layout conflict, no timing. See
SOFTMAX071.md. NSA072 shared staging implemented, syntax PASS;
d64072 seed423 completed exact29/29 PASS; geomean0.963198706x and
cumulative0.960425014x, not promoted. NSA057 remains working candidate.
NSA057 retained. See MASK068.md. Only D64/BS16 simple path changed.

NSA057 G32-only PV FullCol: direct G32 suite7/7 PASS with gains on all5
changed configurations; public057109/109 PASS, geomean1.06897306x.
edges05778/78 PASS, max_abs0.00390625. public057repeat seed408 completed
109/109 PASS, exact coverage, geomean1.06903601x. NSA057 is the working
local candidate based on isolated G32 gains; aggregate improvement over
NSA053 is not established. Preserve both. See FUSED054.md.

NSA053 combines small S8 gather with grid-dependent KD64/KD32 and online
fallback at grid>=512. Full public053 seed398:109/109 PASS, exact coverage,
geomean1.06924303x versus #115804. edges053:70 checks completed, max_abs
0.00390625. Independent public053repeat seed399 completed109/109 PASS,
exact coverage, geomean1.07209337x. NSA053 is the current local candidate;
NSA049 remains preserved. Repeat unchanged-control geomean1.00031975x.
See GATHER050.md and GATHER052.md for direct-parent comparisons and rejected
wide gate. No OJ score beyond the user-reported reference is established.

NSA049 small-grid gather sparse9 PASS: small S4 15.974->12.070us.
Full109 seed390 all PASS, geomean1.06500097x. Small-grid edges completed;
repeat seed391 all109 PASS with exact coverage, geomean1.07012829x.
NSA049 is now the current local candidate; no new OJ score established.
Profile049 fixes small-grid changed/control classification. See GATHER049.md.

NSA048 row reciprocal sparse9 PASS, timings near parent; not promoted.
See GATHER048.md.

NSA047 bounds-only K/V loads sparse9 PASS, timings near parent with no
clear gain; not promoted. See GATHER047.md.

NSA046 S4 KD16 fails GEMM layout inference (Divide by zero); not a
correctness/performance result for changed path. See GATHER046.md.

NSA044 KD32 sparse9 PASS, S4 candidate19.149/57.869us. Direct parent
NSA040 comparison seed383 all9 PASS: S2 regresses, S4 improves
62.426->57.792us. NSA045 S4-only KD32 full109 seed384 all PASS,
geomean1.06226717x. Extended edges completed; repeat seed385 all109 PASS,
geomean1.06203902x. NSA046 S4 KD16 experiment seed386 running.
See GATHER044.md. Not promoted yet.

NSA043 FullCol gather sparse9 PASS, near prior FullRow timings without
clear benefit. Not promoted; see GATHER043.md.

NSA042 gather256 fails GEMM layout inference (Divide by zero), not a
GPU correctness/performance result. Rejected; see GATHER042.md.

NSA041 S8 extension sparse9 PASS but changed cases regress41.306->54.118us
and123.226->209.498us. Rejected; preserve NSA040 S2/S4-only gather.
See GATHER041.md.

NSA040 sparse9 PASS, S4 B2 28.838->19.955us and B4 78.234->62.490us.
Full109 seed377 all PASS, geomean1.06215505x. S4 edges completed;
full repeat seed378 all109 PASS, geomean1.06162384x. NSA041 S8 experiment
seed379 running. See GATHER040.md; no OJ score established.

NSA039 gather128/shared probability sparse9 PASS, larger S2 candidate
12.659us/35.878us. Direct parent NSA034 pairing seed373 all9 PASS:
13.568->12.646us and41.523->35.827us. Full109 seed374 all PASS,
geomean1.05559446x. Edges completed; full repeat seed375 all109 PASS,
geomean1.05699444x. NSA040 S4 gather extension sparse seed376 running;
see GATHER039.md. Not yet promoted.

NSA038 ordinary CK1 both chunk checks PASS but BS16 54.669us vs current
approximately48us. Not promoted; see CK038.md. NSA034 remains baseline.

NSA037 shared vec4 both chunk checks PASS, BS32 63.475us. Neither vec4
nor vec16 improves on current default-layout NSA034. Not promoted;
see LAYOUT037.md. Keep NSA034.

NSA036 shared Q/K/V vec16 both chunk checks PASS, BS32 65.587us vs
NSA034 approximately59us. Not promoted; see LAYOUT036.md.

Newest established local baseline NSA034: two109 PASS runs, geomeans
1.05624632x/1.05717658x, direct parent gains and40 edge checks PASS.
NSA035 QK loop unroll two-case checks PASS; direct parent BS16
48.474->48.026us is inconclusive. Wide31 D128 seed368 all PASS,
geomean1.00314374x with mixed regressions; not promoted.
See UNROLL035.md. No OJ score established.

Current local performance baseline: NSA033, two complete109 PASS runs
(geomean1.05438805x and1.05261737x vs #115804) and direct31-case D128
comparison1.02860972x vs NSA028. No OJ score inferred. NSA034 combines
shared CV2: public109 seed363 all PASS, geomean1.05624632x. Direct31
D128 comparison vs NSA033 seed364 all31 PASS, five changed cases
geomean1.02802231x. Extended40 edges PASS. Full repeat seed365 running;
see COMBINED034.md.
Preserve older candidates; NSA033 remains the established local baseline.

NSA032 ordinary safe-off CV1 direct NSA028 comparison passes both cases:
BS16 50.522->48.230us, BS32 unchanged control61.440->61.478us.
Full public109 seed359 all PASS, geomean1.0506872x, but two ordinary BS32
cases regress. NSA033 gates CV1 to BS16; public109 seed360 all PASS,
geomean1.05438805x vs #115804. Direct31 D128 parent comparison all PASS
seed361, geomean1.02860972x vs NSA028. Independent full109 repeat seed362
running; see GATED033.md. Still provisional, no OJ score established.
See ORDINARY032.md; not yet promoted.

NSA030 shared safe-off chunk CV2 passes two-case checks, including direct
NSA028 pairing: BS32 60.800->59.392us. Full109 public030 seed354 all PASS,
geomean1.0490468x vs #115804. Extended36 edge checks all PASS, max abs
0.00390625. Independent repeat seed355 all109 PASS, geomean1.0453517x.
This overlaps NSA028, so global improvement is not established. NSA031
shared CV1 passes two cases but BS32 slows to67.226us; not promoted.
See SHARED030.md and SHARED031.md.

NSA029 early synchronous V load: all six qk cases pass (seed351), but
D32/BS32 21.901->25.843 us and D64/BS32 47.117->59.942 us regress.
Not promoted. NSA028 remains the best tested combined candidate, not an
OJ-verified score improvement. See EARLY029.md and retained raw results.

NSA022 S1-only safe-memory legalizer disable passes all6 seed333: D128
BS16 55.99->50.10us, BS32 80.10->68.83us. Full public109 seed334 completes:
all109 pass, overall geomean1.017075x, D12831 cases1.094285x. See SAFE022.md.
NSA023 combined six-case test passes seed336. D128 BS16 55.12->50.39us,
BS32 81.63->62.00us. All16 boundary checks pass, max abs0.001953125.
Full public109 seed337 completes all109 PASS: geomean1.030425x, summed
latency ratio1.043529x, changed31 geomean1.118109x. Large-grid boundary
extension passes all20 checks (max abs0.00390625). Independent public109
repeat seed338 completes all109 PASS: overall geomean1.033343x (first1.030425x),
changed31 geomean1.123402x. No OJ score established. NSA024 adds gated S2
gather from NSA002 to NSA023: all9 sparse cases pass seed339. Changed S2
cases19.70->13.48us and50.71->41.41us. Full public109 seed340 all PASS:
overall geomean1.035516x, changed33 geomean1.129247x. NSA025 tests CV2
in ordinary safe-off chunk only (shared path unchanged), seed341 both PASS.
BS16 baseline56.61->49.84us; this is not proof of a gain over NSA024.
Direct NSA024-vs-NSA025 seed342 both PASS: BS16 51.20->50.02us (~2.3%
lower), unchanged BS32 61.52->61.70us. Full public025 seed343 all109 PASS:
geomean1.043981x, changed33 geomean1.147975x. Single-session wider result;
all20 edge checks PASS (max abs0.00390625). Independent repeat seed344
all109 PASS, overall geomean1.042227x vs first1.043981x; changed33
geomean1.153085x. No OJ score established. NSA026 vec4 shared-layout
ablation all6 PASS seed345, but D32/D64 regress7–20%; reject.
NSA027 vecSize16 all6 PASS: D32 gains~3–4%, D64/BS16 regresses. No global
promotion; all10 D32 widening checks pass seed347. Small-grid mixed;
large grids modest wins. NSA028 gates D32 vec16 at B*L*H>=4096 atop NSA025,
full public109 seed348 all109 PASS: geomean1.045651x, summed ratio1.061858x.
Direct NSA025-vs-NSA028 seed349 all10 PASS: four changed large D32 cases
improve~3–4.5%. See PAIRED028.md. Full repeat seed350 all109 PASS,
geomean1.047443x vs first1.045651x. No OJ score established.
Not promoted; see SAFE022.md for aligned/nonnegative-block input assumptions.

NSA021 reuses K storage for V in simplified S1. All6 checks pass, but no
broad improvement: D64 BS16 29.71->30.00us. Compiler-source export confirms
baseline already aliases Qs/Vs; NSA021 does not shrink the data buffers.
See REUSE021.md and source021/ snapshots. Do not merge.

NSA015/016 direct multiply-reduction experiments rejected: NSA015 passes three
cases but is dramatically slower and fails launching D64/BS32; NSA016 shared
probability bridge passes all four cases but remains 13–49x slower. See VECTOR015.md.
NSA013 broader D64/BS32 sweep passes all8 (seed327). Large grids improve
45.13->36.97us and81.75->65.56us; small grids do not consistently improve.
NSA017 combines NSA007 with gated NSA013 D64/D128 BS32 paths. Full public109
passes for both baseline and candidate (seed328): overall geomean1.00816x;
changed5 geomean1.19924x; unchanged104 geomean0.99978x. See PUBLIC017.md.
NSA018 D64 CK2/CV4 passes four checks but regresses changed D64 cases:
BS16 29.76->220.20us, BS32 46.43->109.16us. Reject. NSA019 changes only
D64 CV to2 (output tile32 rather than16): all4 pass, D64 BS16 regresses
29.79->35.00us while BS32 improves46.45->38.25us. Not merged (NSA013 already
has comparable BS32 gains). NSA020 deferred S1 normalization passes all6
but shows no broad gain (BS16 regresses, BS32 D32/64 only~2.5%). Not merged.

NSA014 padding to32 passes but regresses BS16 (56.56->60.70us); reject there.
BS32 gain repeats (~80.20->65.74us). Full-PV CV1 sweep completes all checks;
BS16 best only~1% difference, BS32 regresses. No promotion; see PAD014.md.

NSA012 BS32 compilation fails due to probability-fragment layout conflict.
NSA013 shared probability resolves it and passes six cases. D64/BS32
46.31->37.30us; D128/BS32 79.31->65.51us. D32 regresses; no global promotion.
See WARPS012.md; broader tests needed before combining.

NSA011 cython backend completes four checks but is substantially slower;
 reject. NSA012 tests two-warps for S1 BS32 only, comparison running.

NSA008 QK-fragment larger tests pass but regress most configurations. NSA009
chunk-Q-only passes, regresses BS16 and gives only~5% BS32 benefit. Neither
promoted. Raw numbers in QK008_Q009.md. NSA010 passes with only1-2% timing
differences (not established gains). NSA011 built-in cython backend small-case
comparison running; device math and pass settings identical to #115804.

NSA007 PUBLIC109 COMPLETE: both #115804 and candidate pass all109 with seed316.
Overall geomean1.0070x, changed5 cases1.1227x, unchanged104 controls1.0017x.
Benefit is too narrow to support88-point expectation. Full raw logs and scope
in PUBLIC007.md. NSA008 QK-fragment ablation passes3 smoke shapes/two seeds;
paired larger D32/D64/D128 x BS16/32 testing started (seed317), pending.

NSA004 expanded seed315 test: eight D64/BS32 configurations pass; large work
benefits, small work regresses. NSA006 padded G8 fallback passes12 checks
(D32/64/128 x S1/4 x two seeds). NSA007 combines work-gated optimizations and
fallback. Public109 full entry-point comparison started (seed316), pending.
See QWIDE004_SMALL006.md. No final recommendation or OJ-score claim.

S1 D128 split sweep now has concrete local evidence: CK4/CV4 versus baseline
CK2/CV4 is 69.20 vs79.97 us for B2/L2048/H1/HQ16/BS32, but regresses BS16
(74.21 vs54.60 us). Six non-baseline split combinations plus baseline were
tested across two configurations; all checks passed. See S1_SWEEP.md and raw
rounds under results/2026-09-07. No promotion or OJ-score claim yet.
New-process seed2027 repeat also passes: BS32 baseline80.96 vs CK4/CV4 68.67us.
Experimental NSA003 records the BS32-only rule and passes syntax checking;
full run_kernel testing and wider shape coverage remain next gates.
Update: all eight public D128/S1/BS32 cases pass through both full entry points.
Large grids improve1.08-1.17x; smaller grids are neutral/slower. NSA005 adds a
grid>=2048 gate (derived candidate, not yet independently checked).
Additional G8 input fails in original baseline compilation (M must be divisible
by16); subsequent G32 case was not reached. See COVERAGE003.md for exact scope.
NSA004 independently tests fragment rather than shared Q in simplified S1;
smoke validation passes all three cases/two seeds. Four larger D32/D64 tests
also pass; D64/BS32 improves46.23->41.57us, D64/BS16 slightly regresses.
See QFRAGMENT004.md. Wider coverage/repeat remains needed. No OJ submission made.

User supplied source and identified OJ #115804 (reported score 82.86).
Frozen as `baselines/oj_115804.py`; see `baselines/PROVENANCE.md`.
GPU kernels and shape dispatch match the already tested historical baseline;
only an int32 dtype guard differs. The exact source's historical comments are
not new verification evidence. No claim of reproducing the OJ score locally.

mcProfiler 3.8.1.4 is available and a correctness-checked capture harness has
been added under tools/. Initial global counters are not isolated kernel
evidence. Per-kernel baseline and gather collection both completed and both
passed full correctness checks with seed2718. Counter scope remains uncertain:
achieved-wave counts differ from expected launch geometry and task JSON lacks
UMD metadata. No occupancy/bottleneck conclusion is justified yet. See raw logs
and caveats in results/2026-09-07/profiler/README.md.

## Verified instance

- SSH authenticated with host-key verification; credentials are not stored here.
- C500, sGPU disabled, 65536 MiB in mx-smi, 104 multiprocessors in PyTorch.
- Initial GPU utilization 0%, 826 MiB used, no listed GPU processes.
- Driver 3.8.30; MACA 3.7.1.5; Python 3.12.11.
- PyTorch 2.8.0+metax3.7.1.3.
- TileLang reports 0.1.10+cuda.gitf549117c; imported from /opt/tilelang-metax-v0.1.10.
- Import requires explicit PYTHONPATH and library paths; no package installation or global environment change was made.
- Isolated remote workspace: /root/nsa_20260907_codex.

## Sources and provenance

Public upstream, retrieved on 2026-09-07:

- https://raw.githubusercontent.com/tile-ai/tilelang-metax/race/race_tests/nsa/test_tilelang_nsa_fwd.py
  SHA256 4aeac1535b970e9699e0aae8f2cb261e4340aec898f3fd2b52c48cc66bc6181d
- https://raw.githubusercontent.com/tile-ai/tilelang-metax/race/race_tests/nsa/test_cases_nsa_fwd.json
  SHA256 fa5fb2ec8f0de660bd03b23c743c8c6922c6e2606ca310f71e5ef03586fc8c2f
- Existing user-repository baseline xpuoj_data/nsa_submission.py:
  SHA256 1ff903c9693cdb34487d6860526134af16a1a3135038c880ebc98cc923c89f0f.
  Historical log claims two Accepted submissions at 82.86; not reproduced yet.

The public suite has 109 cases: S=1 in 100, S=2/4/8 in 3 each;
D=32/64/128 in 33/45/31 respectively. All have is_causal=true.
These are public example cases, NOT verified to be the complete current OJ suite.

## Important differences and correctness risks

Update: user supplied the current OJ statement for contest/7/problem/1.
Confirmed scale=1/sqrt(D), causal=1, contiguous FP16 Q/K/V/output and int32 indices,
sentinel=seq_len, allclose rtol=atol=1e-2. B in 1/2/4/8, L=64..8192,
H=1/2, HQ=16/32, D=32/64/128, S=1/2/4/8, BS=16/32.
The reference masks individual token positions (0<=pos<L and pos<=query).
The local reference was updated to exactly this validity condition.
OJ lists PyTorch 2.8.0+metax3.7.1.5; local package reports 3.7.1.3.
The pasted statement supersedes the outstanding scale/interface questions below,
but does not prove the 109 public shapes are the entire OJ suite.

Version clarification: user provided the selected image label
`PyTorch-Agent / 2.8.0 / Python3.12 / maca3.7.1.5` and mx-smi confirms
MACA3.7.1.5. The installed torch.__version__ suffix is a different metadata
field, not proof of the wrong image or an incompatible MACA installation.
No upgrade is indicated solely by these labels.

1. The public standalone test uses scale=0.1; the contest tutorial submission
   defaults to 1/sqrt(D). Exact current OJ semantics still need confirmation.
2. The public standalone test skips reference checks for some large cases.
   A successfully completed public benchmark alone is insufficient evidence.
3. Current public race branch submission.py URL returns 404; the cached official
   tutorial contains a template. Do not silently treat the old tutorial as the current OJ contract.
4. Historical baseline's S=1 paths unconditionally apply causal masking and
   lack a negative block-index guard. Test outside the published all-causal suite
   once current supported input contract is known.
5. Public template uses T.Pipelined(..., num_stages=2). Final-round rule scope
   needs confirmation rather than automatically importing MOE-specific restrictions.
6. Full-device memory capacity does not prove matching clocks, compiler,
   benchmark overhead, distribution, or final OJ ranking.

## Work started

- Independent smoke_nsa.py compares every output element to a gathered sparse
  attention reference, for three small configurations and two fresh random seeds.
- Torch is used only in the test harness/reference/timing, not kernel core math.
- One warmup / one timing per check is exploratory only, not a final ranking.
- Smoke harness passed local Python syntax check and was deployed with a frozen
  copy of the old baseline. Two S=1 cases passed both seeds (maximum absolute
  error <= 0.001586); S=4 compiled but process exited 139 with a segfault.
  No S>1 correctness result was obtained. See smoke_baseline.log.
- NSA001 changes only the S>1 loop from T.Pipelined to T.serial (plus a provenance
  comment). It is a diagnostic candidate, not yet a recommended submission.
  This isolates the pipeline path without changing attention math.
- NSA001 also passed both S=1 cases/seeds and segfaulted after S=4 compilation.
  Therefore replacing the pipeline loop alone did NOT resolve the failure;
  the root cause is still unknown (do not claim asynchronous execution caused it).
  Raw outputs are in smoke_baseline.log and smoke_serial.log.
- CRITICAL CORRECTION after reading the live contract: diagnostic Cython adapter
  reported `Static stride mismatch for parameter BI: expected 8 at index 1, got 1`.
  The smoke harness's sort/transfer path produced NON-CONTIGUOUS indices, contrary
  to the OJ contract. Thus earlier S>1 crashes are INVALID kernel evidence, not
  proof of faulty baseline math or pipeline behavior. The harness now explicitly
  makes indices contiguous and asserts contiguity of all five tensors.
  Default adapter's segfault versus Cython's ValueError is an additional runtime
  error-reporting issue; package version mismatch is not established as its cause.
- NSA002 gather candidate was implemented (one softmax across selected blocks,
  D tiled by at most 64); its first smoke attempt had the same invalid input
  issue and is not a correctness/performance verdict. Corrected reruns pending.
- Corrected frozen-baseline smoke now passes all three shapes and both seeds,
  including S=4/H=2. Maximum FP32-reference absolute error <=0.001586.
  See smoke_fixed_baseline.log. No baseline S>1 math/runtime failure remains
  demonstrated by these tests.
- Corrected NSA002 smoke also passes all three shapes and both seeds. The S>1
  max absolute FP32-reference error is <=0.001491. Single-shot timing is not
  enough to establish a gain. Paired ABBA comparison of the nine public S>1
  cases is now running with ten calls per timed batch and four rounds.
- Paired comparison completed: all nine public S>1 cases passed for BOTH kernels.
  Timing medians in microseconds (baseline / gather), D64 H1 HQ16 BS16:

  | B | L | S | baseline us | gather us |
  |---|---|---|---|---|
  |1|256|2|11.01|11.02|
  |2|512|2|18.82|13.31|
  |4|1024|2|50.39|41.86|
  |1|256|4|15.87|13.44|
  |2|512|4|26.98|25.55|
  |4|1024|4|78.76|82.79|
  |1|256|8|25.00|20.95|
  |2|512|8|39.56|70.50|
  |4|1024|8|121.73|248.88|

  Conclusions: gather is not a universal replacement. Promising S2 improvement
  for larger grids; severe S8 regression at larger grids. Next experiments can
  use shape-based dispatch, but must verify across more seeds/shapes/repeated
  sessions before promotion. These are paired local call-event measurements,
  not OJ scores or proven end-to-end gains. Raw JSONL and full log retained.
- Next: resolve current OJ task contract, reproduce baseline, cover all public
  cases with correctness checks, profile, then compare isolated candidates with
  paired measurements and final repeatability checks before any OJ submission.
