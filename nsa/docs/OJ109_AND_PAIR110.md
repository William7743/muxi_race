# NSA109 OJ confirmation and NSA110 sequential-pair experiment

## Actual OJ result

Read-only detail query confirms NSA109/#141658 Accepted **86.00**. Its full
source SHA equals the tested file. All14 formal points are Accepted and their
checker scores reproduce the displayed total. This supersedes the Pending
snapshot in CALIBRATION_097_103.md; no submission or rejudge was performed.
NSA109 is now the highest directly verified accepted submission in this task,
ahead of v236/#14159485.93, but the88-point objective is not achieved.

Compared with NSA103/#14164885.79, the changed case10 remains8us/88 points.
Cases3,7,14 gain1,1,2 integer points, respectively; case13 loses1. All those
paths have identical normalized local generated CUDA. Thus the net3/14=0.2143
score increase cannot be attributed to the S2 change. OJ binary identity and
the cause of cross-submission variation remain unverified.

The pre-submission seed211 pairing, projected from fixed NSA103 actual times
and disallowing speculative improvements on unchanged paths, predicts85.8571.
Actual86.00 is0.1429 higher. Its case10 projection crosses a score threshold
at a sub-microsecond time change, whereas OJ still reports8us. This illustrates
why integer-microsecond report precision matters; it does not identify the
internal rounding rule and is not a reason to fit a favorable rule.

Evidence: oj109_final.json and oj109_verification.json. Offline verification:
`python nsa/tests/verify_oj109.py`.

## NSA110 design

The previous v133/v134 pair scheme widened the QK/PV tile to32 sparse tokens,
increasing shared memory and register demand and losing performance. NSA110
keeps the original16x64 K/V shared buffer and original GEMM tile, but retains
two separate FP32 score fragments. It performs QK0,QK1, one combined online
maximum/denominator/output rescale, then PV0,PV1. Two score updates share one
normalization state update without materializing a full128-token score table.

The original H1/G16/D64/S8/BS16, L>=1024 and block-aligned dispatch gate is
retained. Invalid pairs are skipped; invalid members remain masked. Each K/V
load and shared reuse is protected by explicit full64-lane warp synchronization.
No asynchronous operations, external compute, input/result caching or custom
compiler options are added. Probabilities and outputs preserve the existing
FP16/FP32 types, though changing the rescale order still requires numeric tests.

`audit_nsa110.py` verifies unchanged inherited AST after removing the one new
factory and its dispatch gate (leading documentation literals are excluded).
It is not a numerical proof or an execution of OJ's language validator.

## Initial runtime evidence

The new factory compiled, and seed0/public plus current distributions passed
their paired FP32 reference checks:4 assertions total,2 per module. However
**0/12 timing samples qualified** under the unchanged before/after drift and
absolute reference rules. Raw public medians were NSA10993.4144us versus
NSA11079.16544us; current medians75.31008 versus76.52352us. These are not
valid performance comparisons and no speedup is claimed.

After the completed timing run, mx-smi showed approximately62% whole-device
utilization while this slice had no visible process. This is consistent with
shared-device interference, not proof of each sample's specific cause. Guard
rejection is retained; no threshold was relaxed to make the candidate pass.

Additional boundary checks and resource inspection are recorded separately;
do not submit NSA110 without qualified independent timing and final validation.

## Completed boundary and independent checks

NSA111 changes only the position of the first valid V block's synchronous
shared load: it occurs after both QK operations but before paired softmax,
instead of immediately before PV0. No other math or synchronization changed;
audit_nsa111.py verifies the exact movement relative to NSA110.

Independent seed137/public+current, reverse initial module order111/110/109,
five alternating rounds:6/6 reference assertions passed, **0/30 timing samples
qualified**. Public raw medians109/110/111 were76.17536/83.45088/85.15072us;
current98.01728/101.18656/83.82976us. They are archived, not ranked.

Stress uses B1/D64/S8/BS16, with L1024/H1 and L1040/H1 testing the new path,
and L1025/H1 and L1024/H2 testing fallbacks. Four in-place input updates cover
random current blocks, only-current-first, only-current-last and Q=0 with
alternating V=+/-1024. Scaled cases use Q/Kx4 and Vx16. All48/48 assertions
passed across109/110/111 (16 each); these are four shapes, not48 distinct shapes.
Together with paired timing prerequisites there are58 source-bound reference
assertions:20 for109,20 for110,18 for111. No full14-point probe promotion is claimed.

Runtime resource queries for case12 both returned success:

| Version | threads | dynamic shared bytes | registers/thread | local bytes | theoretical blocks/SM |
|---|---:|---:|---:|---:|---:|
| NSA109 | 64 | 2560 | 82 | 0 | 20 |
| NSA110 | 64 | 3072 | 80 | 0 | 21 |

The K/V allocation stays16x64, but additional compiler scratch means total
shared memory does not remain2560 bytes. Resource upper bounds are not
achieved occupancy or proof of a bottleneck. No resource claim is made for111.
All jobs in this batch completed; neither110 nor111 is recommended for OJ
without a qualified timing comparison. Keep the verified NSA10986.00 result.

Reproduce source/result audits with `audit_nsa110.py`, `audit_nsa111.py`,
`verify_pair110_111.py`. Result file `pair110_111_verification.json` binds exact
source hashes and expected coverage. `nsa110_111_confirm.jsonl` is one
three-module run; its two comparison summaries must not be double-counted.
