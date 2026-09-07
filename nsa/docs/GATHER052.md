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

paired052 all6 PASS. Direct051->052us:1/64 12.032->12.902;
1/12813.619->14.707;1/25621.504->16.883. Unchanged larger shapes
27.098->27.149,26.931->26.995,40.934->41.050.
NSA053 combines KD64 belowgrid256, KD32 at256..511; online at512+.
Only S8 D64 BS16 G16 affected; inherited S2/S4 unchanged.
Full public053 seed398 launched. No OJ score or full-pass claim yet.

public053 completed109/109 PASS with exact public coverage. Geomean
1.06924303x; summed latency ratio1.08892023x versus frozen reference.
Profile053 uses the corrected512 threshold. Unchanged-control geomean
0.99651780x. Combined edges053 started; OJ improvement remains unverified.

edges053 terminal,70 completed checks, max_abs0.00390625. Raw log retained.
Independent full public053repeat seed399 launched. Do not infer a repeat
pass from first-run or edge results.

public053repeat seed399 completed109/109 PASS, exact coverage. Geomean
1.07209337x, summed latency ratio1.09050183x. Unchanged-control geomean
1.00031975x. With direct-parent evidence and edges, promote NSA053 as
local candidate; preserve NSA049. No OJ88 claim.
