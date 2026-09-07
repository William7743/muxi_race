# Expanded Q-fragment and G8 evidence

## NSA004, fresh process/seed315

D64/S1/BS32/H1. Complete run_kernel calls, full-output FP32 reference checks,
four ABBA rounds of ten calls, current partial blocks included. All eight pass.

| B | L | HQ | baseline us | NSA004 us |
|---|---:|---:|---:|---:|
|1|128|16|12.25|12.33|
|1|512|16|10.76|11.46|
|1|1024|16|13.99|15.39|
|2|512|16|13.76|15.21|
|2|1024|16|25.42|24.33|
|4|1024|16|45.11|41.70|
|8|1024|16|80.68|71.60|
|2|512|32|27.06|21.71|

Supports a work-size gate rather than universal Q fragment. NSA007 proposes
D64/BS32 and B*L*HQ>=32768; this gate covers measured winning configurations,
but extrapolation to other shapes still needs verification. Not an OJ score.
Raw logs/rounds: qwide004.log/jsonl.

## NSA006 padded-head fallback

For G<16 only, use a 16-row TileLang gather kernel, zero-fill padded Q rows,
and never store padded output rows. Existing G>=16 paths are unchanged from
NSA005. No cached outputs or non-TileLang core math.

Test B1/L128/H2/HQ16/BS32, D32/64/128 x S1/4 x seeds10/11:
all12 full-output checks PASS. Sentinel slots and current partial blocks are
included; each query has at least one valid token. This fixes the tested
baseline G8 compile failure. It does not establish all possible G8 cases,
all-masked-row semantics or performance. See small006.log.

## NSA007 combined candidate

Includes NSA005 CK4 grid gate, NSA006 G8 fallback, and work-gated Q fragment.
Python syntax checks pass. Full public109 comparison against #115804 started
with seed316 and fresh current-input reference checks; results pending.
This is not a recommended final submission yet, nor evidence of88 points.
