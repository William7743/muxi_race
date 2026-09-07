# NSA068: post-QK causal mask

Parent: NSA057. Only nsa_simplified with D=64 and block_size=16 changes:
zero-initialize QK accumulation, then replace future-token scores with -inf
after GEMM. Other dispatch, loads, normalization and PV remain unchanged.
Hypothesis: zero initialization may simplify GEMM initialization/scheduling.
This is not an established speedup or OJ score improvement.

Local Python syntax check passed. C500 suite d64bs16 (29 public cases),
seed 419, job d64068 started; results pending. Frozen reference #115804
is the timing baseline. Preserve NSA057 pending measured evidence.

Completed: exact coverage 29/29 PASS. Reference/candidate geomean
0.979076744x, cumulative ratio 0.991047752x (0.913971198 / 0.922227205 ms).
No overall benefit; do not promote. Raw JSONL and compilation log archived.

Next independent NSA069 uses NSA057, not NSA068: D64/BS16 normalized
probabilities written directly into FP16 acc_cast instead of first updating
acc_s then copying. All other paths unchanged. Syntax PASS; d64069 seed420
launched after d64068 terminal. Results pending, no speedup claim.
