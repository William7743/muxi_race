# NSA023 prepared combination (not yet GPU tested)

Parent NSA017 passed public109. NSA022 showed D128 local gains in six-case
screening; its full public109 run is still in progress. These gains cannot
be added arithmetically to NSA017's gains.

NSA023 adds copies of the ordinary and shared-probability chunk functions
with only tl.disable_safe_memory_legalize=True changed. Selection applies
only to S1 D128 with sequence length divisible by block size and G>=16.
G8, simplified, S>1 and other paths retain NSA017 dispatch. Valid block IDs
or the contract sentinel are assumed. Unaligned shapes keep the parent's
ordinary legalizer path; no new general correctness guarantee is implied.

Python syntax check passes. GPU validation is pending, queued conceptually
after public022 finishes (no concurrent benchmark launched). No OJ submission.
