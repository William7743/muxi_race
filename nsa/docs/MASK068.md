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

NSA069 completed exact29/29 PASS, reference/candidate geomean0.996570865x,
cumulative0.999220126x (0.918412799 / 0.919129604 ms). No gain; not promoted.

NSA070 independently branches from057: transpose shared V to [D,BS] only
for D64/BS16 simple kernel, write with T.Parallel and use transpose_B=True
for PV T.gemm. No external computation or cached result. Hypothesis is
different PV operand layout; possible load/store overhead must be measured.
Syntax PASS; d64070 seed421 launched after069 terminal. Results pending.

NSA070 completed29/29 PASS, reference/candidate geomean0.810117694x,
cumulative ratio0.728728112x. Clear regression; do not promote.
Next action: source057simple exports current057 D32/D64 BS16 generated
code for inspection. No further candidate queued before inspection.
