# NSA C500 64 GB baseline setup — 2026-09-07

MOE v748 and its final delivery ZIP remain unchanged. No NSA OJ submission made.

## Latest update: exact reference submission

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
