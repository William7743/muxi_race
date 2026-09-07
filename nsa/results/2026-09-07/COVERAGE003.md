# NSA003 expanded entry-point comparison

Both exact #115804 and NSA003 were called through run_kernel, with seed314,
contiguous tensors, current partial blocks included, full FP32 reference cast
to FP16, atol=rtol=0.01. Four ABBA orders, ten calls per timed batch.
All eight public D128/S1/BS32 configurations passed for both versions.

| B | L | baseline us | NSA003 us |
|---|---:|---:|---:|
|1|512|17.75|18.51|
|2|512|24.90|24.61|
|4|1024|80.64|69.11|
|8|512|80.64|70.58|
|1|1024|24.51|24.64|
|2|1024|44.60|41.23|
|4|512|44.94|40.51|
|8|256|45.09|40.28|

H1/HQ16 for these cases. This confirms a workload-size dependence: use CK4
only at grid B*L*H >=2048 for the next experiment NSA005; keep baseline CK2
below that threshold. NSA005 is a derived candidate, not separately validated
by this run. Approximate speedup range on large grids is1.08-1.17x.
These are local event measurements, not OJ scores.

## GQA correctness gap discovered

The next additional supported-range combination B2/L512/H2/HQ16/D128/S1/BS32
failed to compile in the BASELINE, before candidate execution:
`M must be divisible by 16, but got 8`.
Thus #115804 does not support this G=8 combination in the current environment.
This is a valid contiguous input, unlike the earlier invalid-input diagnostic.
It is not proof that this combination is in OJ's hidden suite. The subsequent
G32 check was not reached because the script stopped on this exception.
Do not describe the whole ten-case run as passing. A padded-head fallback
requires separate implementation and verification.

Raw accepted results: coverage003.jsonl. Full terminal log including the
baseline compiler error: coverage003.log. Prior known-accepted OJ source
remains immutable.
