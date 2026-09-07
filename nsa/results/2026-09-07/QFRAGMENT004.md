# NSA004 Q-fragment ablation

Parent is exact #115804. Only simplified S1 Qs allocation changes from shared
to fragment. The change compiles using normal TileLang T.copy/T.gemm APIs.
No external code, custom device library, or result cache added.

Smoke: three configurations x two seeds pass. Only the D32 S1 smoke case
exercises the changed path; D128 chunk and S4 are unchanged controls.

Paired full run_kernel comparison, seed314, B4/L1024/H1/HQ16/S1:

| D | BS | baseline us | NSA004 us |
|---|---|---:|---:|
|32|16|15.22|14.81|
|32|32|21.63|20.84|
|64|16|29.59|30.30|
|64|32|46.23|41.57|

All four full-output checks PASS, including current partial selected blocks.
Four ABBA orders and ten calls per timed batch; raw rounds retained in
qbench004.jsonl. This is one seed/session and not an OJ-score measurement.
D64/BS32 is promising (~1.11x); D64/BS16 does not support global replacement.
New-seed replication and wider shapes are needed before combining candidates.
