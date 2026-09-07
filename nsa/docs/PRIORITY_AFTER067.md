# Reprioritization after067

Source: public057repeat seed408 local paired JSONL, all109 PASS.
Grouped cumulative candidate milliseconds (rounded):
D64/S1/BS16:29 cases0.91 vs reference0.91;
D128/S1/BS16:23 cases0.60 vs0.69;
D32/S1/BS16:33 cases0.56 vs0.58;
D128/S1/BS32:8 cases0.28 vs0.36.

Prioritize D64 S1 BS16 next: largest cumulative remaining latency with
essentially no gain. Added d64bs16 suite selecting all29 public cases.
G32 micro-tuning has diminishing returns. This is not an OJ weighting or
score estimate. Public file also includes BS64 and L16384 outside pasted
contract ranges; cannot claim exact hidden-test coverage.
