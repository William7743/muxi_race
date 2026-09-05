# v754: independent actual M32 Stage2 source audit

Local read-only analysis of both route dtypes in
`codex_v754_e32_edges_codegen_profile.log`; no GPU execution by this audit.
Log SHA256: `40ece06b5da365e8341619ab2da71144dd72bdfc74cb21e7761fafda29ff4636`.
All line numbers below refer to that log. Complete captured source identities
were independently recomputed, preserving the source's final newline:

| Source | Characters | SHA256 |
| --- | ---: | --- |
| v748 FP32 route | 32767 | `e21838abbd48981aa3064ab3418cf4a7939b9bffeed171e08d5baea364c1165c` |
| v754 FP32 route | 46299 | `3fd8f252d296c37818be4a032f3a4d53a079fbfc884af66d403a5082be3030bf` |
| v748 FP16 route | 33336 | `7a6be56dfc1c529fb62dc694011e58fb889107563a08e84b47f0cbd996483eeb` |
| v754 FP16 route | 46915 | `fb7d1dbf3ec0be273295ea41c4bdd86c6297004ef209d1f88534ce1fc2f623a8` |

## Existing branches remain intact

The entire M128 brace block is byte-equal after indentation trimming, separately
for each dtype: baseline48 vs candidate523, and1173 vs1657. The entire M64 block
is likewise equal (268 vs743;1399 vs1883), except its outer threshold changes
from `>0` to `>32`. No temporary/array renaming is needed for these comparisons;
their loads, MMA, barriers, route expressions and output stores are unchanged.

Nested CTA-metadata-only choices are M128 for rows>64, M64 for33..64, M32
for1..32 (949 /2092), otherwise the existing all-zero output path. They are
mutually exclusive and uniform across threads. FP32 M32 computation949–1095
and FP16 computation2092–2238 are also byte-equal after indentation trimming;
their differences occur only in the route dtype/epilogue conversion.

## Actual M32 layout and complete output

Shared remains Down at byte0 and Up at byte16384 (497–498 /1631–1632),32KiB.
M32 declarations are C float[16], A half[4], two B half[16] (507–510 /
1641–1644). These are logical source arrays, not physical register counts.

Actual expressions were enumerated on CPU over all256 threads/components:

- All eight M32 A LDS sites, four steady966/986/1001/1016 and four terminal
  1037/1057/1072/1087, read exactly shared half offsets0..2047. All eight B
  sites970/975/990/1005 and1041/1046/1061/1076 cover0..8191. They match the
  official swizzled logical32x64 A and128x64 B maps. The FP16 computation is
  identical, so no dtype-specific LDS change exists.
- Prologue global copies957/961 load K0. Next-tile copies1028/1032 load K+1;
  evaluating K=0 and30, bx35/by55/expert31 reproduces the expected bounded
  input/weight coordinates through final K31. A copies exactly32 rows, B128.
- Tiny output1114 /2257 maps C slots0..15 per thread to4096 distinct32x128
  coordinates, with the unchanged external bx128 stride. Its row predicate
  admits valid rows and writes zero for other rows inside0..31.
- The separate uint4 zero loop1117–1119 /2260–2262 has **six** iterations.
  Enumerated coordinates are exactly rows32..127 x128 columns:12288 half-elements,
  disjoint from the tiny epilogue, together covering the full128x128 output.
  It does not make the erroneous32..63-only clear. Empty-block stores1125 /
  2268 independently cover the full128x128 tile.

Bounds assume the valid2373-raw/4608-padded edge metadata, not malformed inputs.
The companion `audit_m32_layout_cpu.py` separately replays official API maps;
the checks here use the actual generated addresses.

## Raw-route clamp, K order and synchronization

Every scalar route reference in each candidate SOURCE (16 occurrences per
dtype, including the M128/M64 paths) retains `max(0,min(index,2372))`.
Tiny references are1102 /2245. CPU evaluation includes the last expert's
raw_start2372/padded_start4480, checking all thread rows remain within0..2372.
Valid rows retain their original raw index; this clamp is not a solution for
a zero-length route tensor, which requires the separate host zero-route path.

Steady tiny MMA order is A0/B0, A1/B1, A2/reloaded-B0, A3/reloaded-B1
(980/995/1010/1020); terminal1051/1066/1081/1091 repeats it. Over31 steady K64
tiles plus K31 terminal, each output accumulates32x4=128 ascending K16 steps.
Each B buffer is overwritten only after its prior value's MMA use; A is freshly
loaded for each microstep. No missing/duplicated terminal tile was found.

| Branch | FP32 steady top/end; terminal | FP16 steady top/end; terminal |
| --- | --- | --- |
| M128 | 538 /615;625 | 1672 /1749;1759 |
| M64 | 758 /835;845 | 1898 /1975;1985 |
| M32 | 964 /1025;1035 | 2107 /2168;2178 |

Top/terminal barriers protect copied Up/Down before reads; steady end barriers
protect those reads before next-K overwrites. There are three static sites per
branch, nine total, but only one branch executes: positive CTAs make63
source-level barrier calls (`31*2+1`), not189. Empty blocks make none. No late
terminal barrier is required because there is no subsequent shared overwrite.

No new generated-source address, fragment, synchronization or output-coverage
defect was found. The raw log separately reports four Stage2-only comparisons
with finite outputs, max_abs_full0, nonzero_diff0, bitwise_equal=True and
padding_nonzero0 (1134–1135 /2277–2278). Those are the recorded edge fixture's
results, not a whole-entry, independent mathematical/OJ or performance proof.
No physical register, occupancy or spill conclusion is inferred from arrays or
private-memory metadata.
