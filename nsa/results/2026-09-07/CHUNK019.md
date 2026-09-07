# NSA019 D64 CV2, seed330

All four qfragment cases pass reference for both exact #115804 and NSA019.
D32 cases are unchanged controls, D64 is the changed path. B4 L1024 H1 HQ16 S1.
D64 BS16 baseline29.7856us / candidate34.9952us; BS32 46.4512 / 38.2464us.
Raw chunk019.jsonl/log. Paired ABBA four rounds, ten calls per timed batch.

CV2 avoids the extreme CV4 regression but still loses on BS16. Do not merge
globally. BS32 gain is similar to an already-tested NSA013 path; cross-session
numbers do not prove which candidate is faster. No OJ score inferred.

NSA020 next tests deferred normalization on both original S1 routines:
retain max-subtracted exp2 and FP32 sum, cast unnormalized exponentials to
FP16, compute PV, then divide FP32 output by the sum. This is mathematically
equivalent but not bitwise equivalent because FP16 rounding moves. Hypothesis:
remove normalization from the pre-PV dependency chain. It does not assume
input values or reuse results. QK, causal mask and S>1 path unchanged.
Python syntax check passed; six-case comparison seed331 running.
