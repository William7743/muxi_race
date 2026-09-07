# NSA068: post-QK causal mask

Parent: NSA057. Only nsa_simplified with D=64 and block_size=16 changes:
zero-initialize QK accumulation, then replace future-token scores with -inf
after GEMM. Other dispatch, loads, normalization and PV remain unchanged.
Hypothesis: zero initialization may simplify GEMM initialization/scheduling.
This is not an established speedup or OJ score improvement.

Local Python syntax check passed. C500 suite d64bs16 (29 public cases),
seed 419, job d64068 started; results pending. Frozen reference #115804
is the timing baseline. Preserve NSA057 pending measured evidence.
