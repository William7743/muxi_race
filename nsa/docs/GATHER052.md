# S8 gate and KD32 follow-up

NSA051 small051 seed395 all6 PASS versus frozen baseline. B/L timings us:
1/64:16.026->11.750;1/128:24.627->13.466;1/256:24.768->21.606;
1/512:27.098->27.238;2/256:26.982->27.059;2/512:41.088->41.139.
Gate correction removes prior512-grid regression while preserving small wins.

NSA052 parent051 changes only S8 gather KD64 to KD32. Hypothesis: lower
shared/fragment footprint may help despite extra GEMM loops, as in S4.
small052 seed396 launched; no performance claim yet. Original candidates
remain intact. Local speedups are not OJ scores.

small052 seed396 all6 PASS. Candidate us by B/L:
1/64:13.043;1/128:14.758;1/256:16.947;1/512:27.238;
2/256:27.021;2/512:41.165. Compared with prior051 measurements,
KD32 appears slower at64/128 but faster at256. Different-run comparison
is preliminary. Direct parent051 paired052 seed397 launched to confirm.
