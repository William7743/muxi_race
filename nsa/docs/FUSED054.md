# NSA054 fused probability store

Parent NSA053. Only nsa_shared_simplified changes: directly store
acc_s/sm into shared acc_cast in T.Parallel instead of updating acc_s
then T.copy to acc_cast. Same normalization and FP16 conversion, no
result reuse. Hypothesis: fewer intermediate operations; compiler may
already fuse the original, so speedup is unproven.

qwide054 seed400 launched after confirming no active jobs. Candidate
is experimental, not promoted. Full correctness/performance pending.
