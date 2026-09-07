# NSA050: small-grid S8 gather

Parent NSA049. Only dispatch change: S8 D64 BS16 G16 uses gather when
B * seq_len * H < 1024. Larger S8 retains online softmax because NSA041
showed regressions there. Hypothesis: reduced loop overhead may help
small grids despite the larger gather footprint. No cached results.

Python syntax check PASS. Remote paired sparse suite seed392 started as
sparse050 and its live channel was confirmed nonterminal. Performance
and correctness results were pending at launch; NSA049 remains the local candidate.

Completed sparse050: all9 PASS. Newly changed B1 L256 S8:24.858->21.709us
versus frozen reference (also the parent's unchanged online path). Large S8
controls40.986->41.037us and123.085->123.046us. This is one local run,
not a confirmed OJ gain. public050 terminated at argument parsing: invalid
suite name public109. No GPU test occurred in that job. Correct suite public
seed393 relaunched with unique label public050fixed. Added small S8 edge
shapes to validate_edges for subsequent execution.

public050fixed seed393 completed:109/109 PASS, exact public coverage.
Geomean1.06669631x, summed latency ratio1.08615055x versus #115804.
Profile050 correctly partitions46 changed and63 unchanged cases; unchanged
geomean0.99833498x. This does not establish an overall win over NSA049's
two runs. Extended edges launched as edges050; retain NSA049 pending checks.

edges050 completed:70 PASS, max_abs0.00390625 across declared cases/modes.
No all-masked semantics claim. Direct NSA049/050 paired test seed394 started
as paired050 with small_s8 suite: B/L=(1/64,1/128,1/256,1/512,2/256,2/512).
Last shape is an unchanged threshold control. This checks the dispatch gate
beyond the single public small S8 case; results pending.
