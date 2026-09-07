# NSA007 complete public109 comparison

Exact #115804 versus combined NSA007. Seed316, complete run_kernel paths,
all109 public cases, full-output FP32 reference cast to FP16. Four ABBA
orders of ten calls per timed batch. Inputs include current partial blocks
and sentinel slots; no cross-invocation result reuse. Both pass all109.
The source testcase multiset exactly matches109 observed PASS rows, with no
missing or extra occurrences. Raw logs: public007.log and public007.jsonl.

| Scope | Cases | Geometric mean speedup | Summed latency ratio |
|---|---:|---:|---:|
|All public cases|109|1.0070x|1.0112x|
|Changed NSA007 branches|5|1.1227x|1.1269x|
|Unchanged controls|104|1.0017x|1.0002x|

Only five public cases exercise the changes. Local improvements are real in
those configurations but too narrow for a large overall gain. Do not equate
this with OJ score, hidden-suite coverage, or achievement of88 points.
Unchanged controls range0.9585x-1.1084x: small individual timing differences
must not be overinterpreted. A second full-session repeat is still desirable.

Reproduce summary with:
`python nsa/tools/summarize_comparison.py nsa/results/2026-09-07/public007.jsonl`.
Next experiment NSA008 moves both QK operands into fragments on S1 paths,
targeting the much larger set of unchanged configurations. It is an independent
ablation, not automatically merged with NSA007.

NSA008 smoke now passes three cases/two seeds. Both changed S1 paths are
exercised; unchanged S4 is a control. Six larger paired cases are pending,
so this does not yet establish a performance improvement.
