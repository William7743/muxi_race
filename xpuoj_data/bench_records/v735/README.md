# v735: Gate/Up 32-column interleaving with 2x2 warps

## Design and identity

Independent E32/H7168/I2048 Stage1 variation of v731, keeping
M128/totalN128/output64/K64, 256 threads, one C accumulator and 32 KiB shared.
The emitter changes from 4x1 warps / warp32x128 to **2x2 / warp64x64**.
Physical Bshared segments are Gate0:32, Up0:32, Gate32:64, Up32:64;
synchronous copy order is Gate0, Gate1, Input, Up0, Up1. The official
lane/local epilogue pairs Gate and Up at local offset +8 rather than +16.

The K16 and epilogue loops remain serial, without stacking the v733/v734
unroll change. Math, passes, swizzle, outer K/terminal structure, Stage2,
dispatcher and other input-shape paths stay v731. No extra launch, global
packing workspace, async/BSM, external device code or result reuse is added.

| Evidence | SHA256 |
| --- | --- |
| Tested v735 Python source | `7ea5bfcfb07edaa91741faf2644596b7fc636ce5e0d8ec6c8068ad6279995136` |
| v735 generated source, 10638 characters | `8e64f732cc302616c3f5ae686ce07c8fa5d8ebfff972452015e20ba6fbc78a53` |
| Raw random-entry log | `848873be639b7bc048dc122f66ff72736dd1313984d9ca51a40bf58dc107ed0d` |

After measurement, result comments and the final newline were updated. The
final repository v735 SHA256 is
`9f7101d2a608da88406658f031e910692a0579ab0c367e402a4e578b569837da`.
Restoring only those non-executable changes reproduces the tested source hash
above; the executable AST remains identical.

The server's comparison v731 remains the original tested source
`f1271c866444c1b9921b721f47c4b6f7ba16b95e33f377669d210578296f4f5c`.
The repository's v731 is
`4f5f160cb39a4d9f91c6f61f877a9ea2fa2808a24dbe3682ca8bbde075ba66f5`
after result-only header/newline updates. Their executable ASTs are identical;
the changed documentation hash does not identify a different measured kernel.

Compilation, Python/Ruff, whole-source/AST isolation and CPU geometry/host
audits passed. [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md) records actual captured
copy/LDS/output address replay and all three required barrier sites.
Registers/spills were unavailable in codegen metadata; logical arrays and
shared-read counts are not measured occupancy or physical register counts.

## Random correctness and all four entry samples

Raw evidence:
[codex_e32_720_731_735_random_entry.log](codex_e32_720_731_735_random_entry.log).
Local E32/H7168/I2048, alternating64-220 routing, 4544 raw and 6144 padded rows,
48 M blocks; fixed random seed 20260903, FP16 input/matrix weights and FP32
route weights. This is not verified OJ input data.

Every candidate passed three repetitions of both `launch_full` and real
`run_kernel`, with workspace/output filled with NaNs before each recomputation:
all comparisons reported max_abs=0.000000 and bad=0/44040192 against freshly
computed c0/v720 output. These are three recomputations of **one fixed random
input set**, not three independent seeds, not NaN input tensors, and not an
independent mathematical reference. The checker uses `abs(diff) <= 0.05 +
0.05*abs(reference)` plus finiteness checks; `max_abs` is printed to six decimal
places. No additional bitwise equality check was performed, so a printed
`0.000000` is not a strict bitwise-equality assertion.

Untraced entry timing used one complete warmup and one timed call per round;
each entry launches two kernels. Four rounds alternate forward/reverse order.
All values below are milliseconds from this single cohort.

| Round | Order | v720 | v731 | v735 |
| --- | --- | ---: | ---: | ---: |
| 1 | forward | 4.663040 | 6.790400 | 6.448640 |
| 2 | reverse | 4.657664 | 7.062528 | 6.662912 |
| 3 | forward | 4.676352 | 6.935552 | 6.664448 |
| 4 | reverse | 4.678912 | 6.794496 | 6.659584 |
| Median | | 4.669696 | 6.865024 | 6.661248 |
| Mean | | 4.668992 | 6.895744 | 6.608896 |
| Population stdev | | 0.008893 | 0.112639 | 0.092541 |

Paired delta = same-round candidate minus comparator, in microseconds.
Positive means slower.

| Comparison | All round deltas | Median delta | Mean delta | Wins / ties / losses |
| --- | --- | ---: | ---: | --- |
| v731 minus v720 | +2127.360, +2404.864, +2259.200, +2115.584 | +2193.280 | +2226.752 | 0 / 0 / 4 |
| v735 minus v720 | +1785.600, +2005.248, +1988.096, +1980.672 | +1984.384 | +1939.904 | 0 / 0 / 4 |
| v735 minus v731 | -341.760, -399.616, -271.104, -134.912 | -306.432 | -286.848 | 4 / 0 / 0 |

The first two paired rows are printed in the raw log; the third is recomputed
from its same-round samples. Median differences and paired-median differences
are distinct statistics and are not substituted for one another.

## Decision

v735 is 2.968% lower median latency than v731 in this cohort, but still
**42.648% higher latency than v720**, losing to v720 in every round.
The helper's printed speedup_vs_c0=-29.898% uses base/candidate-1; it is not
the percentage increase in latency.

**Do not recommend v735 for OJ. Close this branch without extending it to
K32 or N16 interleaving.** The 2x2 remapping improves an already-slower
concatenation implementation, not the verified baseline. These data do not
establish why the structure is slow or predict an OJ score. v720 remains the
verified 80.33-point baseline; v735 has no OJ submission/result.
