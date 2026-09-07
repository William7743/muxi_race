# NSA050: small-grid S8 gather

Parent NSA049. Only dispatch change: S8 D64 BS16 G16 uses gather when
B * seq_len * H < 1024. Larger S8 retains online softmax because NSA041
showed regressions there. Hypothesis: reduced loop overhead may help
small grids despite the larger gather footprint. No cached results.

Python syntax check PASS. Remote paired sparse suite seed392 started as
sparse050 and its live channel was confirmed nonterminal. Performance
and correctness results remain pending; NSA049 remains the local candidate.
