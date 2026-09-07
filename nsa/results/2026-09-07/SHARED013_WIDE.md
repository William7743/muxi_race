# NSA013 wider D64/BS32 comparison

Seed327, H1 D64 S1 BS32. All eight cases pass full reference for both
baseline #115804 and NSA013. Same ABBA ten-call timing harness as prior work.

| B | L | HQ | baseline us | NSA013 us |
|---|---|---|---|---|
|1|128|16|10.42|10.73|
|1|512|16|10.92|11.01|
|1|1024|16|14.25|12.99|
|2|512|16|13.96|13.32|
|2|1024|16|25.61|21.44|
|4|1024|16|45.13|36.97|
|8|1024|16|81.75|65.56|
|2|512|32|27.65|19.33|

Raw: shared013wide.jsonl and shared013wide.log. Local results only.

NSA017 uses NSA007 as its parent and adds the exact NSA013 shared-probability
functions. D64 BS32 dispatch requires B*L*HQ>=32768; D128 BS32 requires
B*L*H>=2048. G<16 fallback has priority. Other paths remain NSA007, no
output caching or phase-dependent behavior. Conservative gates avoid small-grid
regressions; their hidden-OJ applicability is unverified. Full public109
comparison launched seed328. Python syntax check passes; GPU outcome pending.
