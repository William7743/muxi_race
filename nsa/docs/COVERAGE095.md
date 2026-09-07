# Calibration audit after OJ141047

Public109 contains100 S1 cases. Its9 sparse cases all use D64, H1,
HQ16, BS16 (S2/4/8, three batch/length choices). Thus public109 aggregate
does not characterize other contract-allowed sparse dimensions/layouts.
This is a verified coverage limitation, NOT proof it caused OJ67.36.
Exact141047 source and breakdown remain unconfirmed.

New sparse_coverage suite: B2/L512, (H,HQ)=(1,16),(1,32),(2,32),
D32/64/128, S2/4/8, BS16/32 =54 cases. Same current-input reference,
contiguity checks and ABBA timing. G8 excluded from paired frozen-baseline
test due to its known layout issue; not claimed fully exhaustive.
Compare078 directly to frozen115804, seed447, label coverage095.
No new submission kernel. Completed54/54 PASS; exact54 shape keys verified
against planned Cartesian set with no missing or duplicate cases.
Geometric115804/078 ratio1.139740; sum-time ratio1.145490. Worst ratio
0.977365 (D128,S2,BS32,H1/HQ16): roughly2.3% slower, not a broad
large regression. This added slice does NOT reproduce OJ's apparent
score decline and does not identify its cause. It covers only B2/L512,
so other shapes/input distributions and timing semantics remain open.
Do not convert these ratios to scores. Raw coverage095 logs archived.
