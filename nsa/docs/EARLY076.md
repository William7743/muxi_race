# NSA076 early V fragment

Parent075, independent experimental path (not a promoted baseline). Move
V fragment copy before QK while preserving math and D64 BS16 dispatch.
Hypothesis: overlap V load latency with QK, at expense of longer register
liveness. No async API or external code.

Syntax and representative source export PASS. D64 source has V load line44
before QK MMA line69; still3 barriers. Thus scheduling changes at source
level, though machine-level overlap remains unproven. d64076 seed427
launched, correctness/timing pending. NSA057 retained.

Completed exact29/29 PASS. Reference/candidate geomean0.786400613x,
cumulative0.689504406x; rejected. Early fragment load does not recover
the lost performance. Do not infer measured overlap or register pressure
from source order alone. Raw JSONL and compilation log archived.
