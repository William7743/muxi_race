# NSA054 fused probability store

Parent NSA053. Only nsa_shared_simplified changes: directly store
acc_s/sm into shared acc_cast in T.Parallel instead of updating acc_s
then T.copy to acc_cast. Same normalization and FP16 conversion, no
result reuse. Hypothesis: fewer intermediate operations; compiler may
already fuse the original, so speedup is unproven.

qwide054 seed400 launched after confirming no active jobs. Candidate
is experimental, not promoted. Full correctness/performance pending.

qwide054 seed400 all8 PASS versus frozen reference. Candidate us in suite
order:10.381,10.726,14.054,13.850,21.376,37.747,65.344,19.341.
Reference comparison includes inherited gains, not proof of fusion benefit.
Direct parent NSA053 paired054 seed401 launched on qwide to isolate change.
