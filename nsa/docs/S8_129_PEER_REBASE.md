# NSA129: peer343 S8 prefetch on OJ109

## Decision

Retain peer343 byte-for-byte as probe_nsa129_oj109_s8_packed_prefetch.py.
Do not change its historical header: keeping the exact bytes lets existing
peer GPU records bind directly to the source. It is a possible small-gain
manual OJ trial, not a verified score upgrade or88-point solution.
No new OJ submission or GPU test was run by this task.

Source SHA256:
d91a4b0a7709c7a6a795d79a7b0c1b87d94861ee979a91ed875626cd788cdbd4.
Parent109 SHA256:
7abe284475520edf64999c583148cde45cc1c8f0a017503cc98068e7f33986fa.

The peer v343 delta does not fix the S8 body of342: it rebases that same body
onto our actual OJ #141658 baseline. audit_nsa129.py proves whole-module AST
equality with109 after substituting only nsa_online_direct_output from128,
and checks the exact peer343 bytes. The immutable parent remains untouched.
No D128 custom compiler flags or NSA127 bounded-load combination is present.
The minidiff is results/2026-09-09/nsa129_vs109.diff.

## Source-bound peer results

| Recorded run | Coverage | Qualified timing samples |
|---|---|---:|
|Full suite, seed0/public|14 cases x3 modules =42 reference assertions|210/210|
|S8 confirmation, seeds42/137, public/current|1 case x2 modules x4 input sets =8 reference assertions|56/56|
|Value stress, seed137|12 source-bound assertions, parent+candidate|Not used for timing|
|Current-block causal boundaries|2 source-bound contract-compatible assertions|Not used for timing|

The42 full-suite assertions include14 for the candidate,14 for parent109 and
14 for historical control176. Do not call these42 candidate tests. Guard
qualification is recomputed from both drift<=3% and control-reference ratio
within[0.92,1.08]. No fitted OJ timing corrections are applied.

All14 generated-source hashes are compared:13 unchanged, only case12 changes.
S8 confirmation guard-relative geometric time ratio is0.9665751163
(about3.34% faster than109). Separate full-suite case12 ratio is0.9683946096.
Timing differences on the unchanged13 paths are NOT optimization benefits.
The earlier3.73% figure was342 versus318, a different comparison.

Using actual109 OJ checker baselines and changing only case12 by the measured
confirmation ratio still predicts case12=82 and overall86.00. This is merely
a fixed-anchor projection on a16GB/25%-compute slice, not an OJ measurement,
not evidence of full64GB performance, and not the88 target.

## Device-source inspection

peer_v343_case12_generated.cu hashes to the exact generated-source fingerprint
in the paired records (b6617d2506adae51fa29fc7496e88058b940761b645095796df1dd496c3b3493).
It shows eight uint registers per lane, two128-bit V loads, paired-half
unpacking and128-bit shared writes. The load occurs before QK MMA. The
existing warp fence after QK precedes shared-buffer overwrite, and the next
warp fence precedes PV shared reads; all four original warp fences remain.
It is generated inspection evidence, NOT source to submit or external code
injected into TileLang. No additional async operation is introduced.

## Limitations and reproducibility

The full-suite paired run uses one seed/public masks, supplemented by two
seeds/two mask modes for S8. We inspected the peer calibration driver:
fresh inputs, NaN-filled output before correctness, FP32 reference allclose
atol=rtol=.01, alternating timing order, and per-sample controls. Raw records
and generated source are preserved; this task did not independently rerun
the GPU workload. No claim of universal correctness or race-sanitizer proof.

Six current-mask diagnostic rows use negative/int64 indices or noncausal
inputs. They are preserved but excluded from contract correctness counts.
Other peer boundary/shape logs lacked source hashes and were not promoted
into source-bound certification. The interrupted original current-mask file
was missing; only the completed current_resume file is used, not an invented
result for the failed attempt.

CPU-only reproducibility, from repository root:

```text
python nsa/tests/audit_nsa129.py
python nsa/tests/verify_nsa129_peer_results.py
```

The verifier checks source hashes, exact case/module/seed/mask coverage,
numeric pass fields, both timing gates, generated-source scope and fixed OJ
projection; it writes nsa129_peer_verification.json and the minimal diff.
NSA127/128 still need their own combination validation. Do not transfer this
peer evidence to a different candidate or claim it completes those tests.
