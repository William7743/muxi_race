# v745: independent runtime-M64 Stage1 source/codegen audit

Scope: local, read-only source analysis plus the archived E32/H7168/I2048
Stage1 generated C++ in `codex_e32_743_745_stage1_codegen.log`. Line numbers
below refer to that log unless explicitly identified as Python lines. This
document does not report GPU numerical results, timing, ISA or hardware counters.

## Source isolation and branch contract

The independent whole-module AST reconstruction used v743's **actually selected**
`_moe_stage1_prefetch_giu_merge`, not its retained `_v527` builder. All original
functions except `_get_stage1` are source-text identical between v743 and v745.
The complete AST permits only the new builder, its M64 branch transformation,
and the narrow Stage1 selector. The full branch is an exact AST copy of the
original two clears, GIU steady loop, terminal K and valid-row SwiGLU epilogue.
The tail branch changes only A copy/view height, independent C names and the
epilogue row count. Pass settings, copy hints and mathematical expressions remain
unchanged. A 36-combination host-selector check confirmed the new builder is
selected only for E32/H7168/I2048 with positive padded tokens and block count.

`actual_rows` depends only on the CTA's block/expert metadata. Full executes for
rows >64, tail for 0<rows<=64, and neither for zero rows. Thus barriers within
either branch are uniform across the CTA. Every row count 0..128 was checked for
exact source-level valid workspace coordinate coverage, without empty/padding
writes. Stage2, raw-weight clamps and host empty handling remain v743: zero
padded rows return without a launch; zero total valid tokens use the existing
one-launch zero-output path; normal calls retain two launches.

Python source hashes at audit time:

```text
v743 5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9
v745 12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96
```

## Actual generated M64 view and independent C layouts

v745 source occupies log lines 223–622. The full branch starts at 247; the tail
branch starts at 433. The shared allocation remains weight at byte offset 0 and
input at 16384 (234–235), each 16 KiB, total 32 KiB. The M64 path does **not**
reduce the physical shared allocation to 24 KiB.

The A `BufferRegion` is genuinely lowered as M64:

- Tail steady/terminal A copies (451 / 511) cover exactly 64x64 halves and shared
  offsets 0..4095. The full branch copies 128x64 halves (265 / 325).
- All four tail A LDS sites (462, 488, 522, 548) read only those first 4096 half
  entries, never the uninitialized upper 64 rows. Their local arrays are eight
  halves, whereas full-path A arrays have sixteen halves. B remains sixteen
  halves in both paths.
- Full Gate/Up C are separate `float[64]` declarations (236–237), and tail
  Gate/Up C are separate `float[32]` declarations (239–240). The tail is a 2x2
  warp arrangement with 32x64 output per warp, not an accidental M128 GEMM.
  Full remains 2x2 with 64x64 output per warp.

Captured global/shared address expressions were evaluated for all 256 threads,
the full vector widths, steady k=37 and terminal k=111, bx35/by15/expert31.
Both tail A copies produce 4096 unique correct shared half addresses; both tail
Gate copies produce 8192. All four A and all four B LDS expressions were checked
over ki=0..3 and their complete local ranges against the same swizzled mapping:

```text
shared_half_offset = row*64 + ((col//4 ^ row%16)*4) + col%4
tail A row = ((thread & 127)>>6)*32 + row_tile*16 + (thread & 15)
B row      = (thread>>7)*64 + col_tile*16 + (thread & 15)
K column   = ki*16 + ((thread & 63)>>4)*4 + vector_component
```

Gate/Up copies still refer to the current K tile. Current Up is read into the
private `up_prefetch[32]` before Gate GEMM, then copied into the reused weight
shared buffer after Gate reads finish. No prior-call value is retained.
The steady global K ranges are k*64 through k*64+63 for k=0..110; the terminal
uses 7104..7167. Gate and Up each accumulate all 448 K16 steps in ascending
order, with no missing or duplicated terminal tile. The tail MMA row loop is
two iterations rather than four, but the ki=0..3 sequence and per-output
accumulation order are unchanged.

All 66 generated assignment statements in the full branch match the actual
v743 baseline after trimming indentation. This includes loads, MMA value
statements, SwiGLU operations and output addresses. Control-flow wrapping is
different because the full path now has an outer rows>64 guard; this is **not**
a claim that the entire generated kernel is byte-equal.

## Actual synchronization

| Scope | Copies -> Gate reads | Gate reads -> Up shared overwrite | Up writes -> Up reads | Up reads -> next Gate/A overwrite |
| --- | --- | --- | --- | --- |
| Full steady | 273 | 291 and 292 | 299 | 317 |
| Full terminal | 333 | 351 and 352 | 359 | No next tile |
| Tail steady | 459 | 477 and 478 | 485 | 503 |
| Tail terminal | 519 | 537 and 538 | 545 | No next tile |

All input/weight producer-consumer transitions are protected in the captured
C++; no automatic barrier is merely assumed from the Python `T.gemm` calls.
The redundant adjacent Gate-to-Up barriers are inherited, not removed by this
probe. Nine static sites occur in each branch; only one branch can execute.
For positive rows and 112 K64 tiles, either branch makes `5*111+4 = 559` barrier
calls according to the generated C++ control flow, not 1118. Empty rows skip
both branches entirely. These are source-level counts, not physical ISA or
hardware measurements.

## SwiGLU and workspace output coverage

Full output is stored at 429 under the row predicate at 379; tail output is
stored at 615 under predicate 565. The complete full/tail SwiGLU FP32 value
expression and final `__float22half2_rn` conversion sequences are equal after
consistent temporary/C-name renaming. This establishes expression preservation,
not an unperformed GPU bitwise comparison.

All 256 threads and every vector component were enumerated using the actual
generated store addresses. Full produces exactly 128x128 distinct coordinates,
tail exactly 64x128. For local loop index i and vector component v=0..3:

```text
row = ((thread & 127)>>6)*(M//2) + (i>>2)*16 + (thread & 15)
col = (thread>>7)*64 + (i & 3)*16 + ((thread & 63)>>4)*4 + v
C local slot = i*4 + v
```

M128 uses i=0..15, slots 0..63; M64 uses i=0..7, slots 0..31. The two C buffers
in each branch use the same row/column mapping. For all applicable actual row
counts, the predicate admits exactly `actual_rows*128` distinct workspace
writes and excludes padding. Output offsets remain `bx*128`, even in M64 mode;
the physical row-block mapping is not changed to bx*64.

Generated hashes were independently recomputed from complete SOURCE bodies:

```text
v743 12945 chars 32f3b1b3e9da925514a1aa02c6313d4bcd34adae330783fdc82e4f4a36f2692b
v745 25102 chars 67282b935424a980038cc9beed0de38dedf723278f7c1c8147645f9d69456782
```

Codegen metadata reports register/spill fields as null. C++ array sizes are not
physical register counts or evidence of no spill. No new source/codegen
address or synchronization defect was found in this scope.

## Read-only review of `remote_v745_stage1_edges.py`

The script is diagnostic-only. Its imported pure metadata function was executed
on CPU: 32 experts, 2373 raw rows, 4608 padded rows, 36 row blocks and 2235 invalid
rows. The masks partition all padded rows, the block map reconstructs every
actual-row count, and rows 0/1/63/64/65/127/128 are covered. An allocated empty
CTA is intentional. The final expert has one valid row followed by 127 padding
rows, so `out[-128]` is correctly the final valid token.

Input initialization and call review found no concrete defect:

- Fixed Gate/Up/Down seeds 74510/74511/74512 are used once. Repeat seeds
  `74501+repeat` generate fresh x and route data; weights remain shared across
  repeats. This is not a claim of fresh matrix weights every repeat. The two
  route dtypes use the same underlying route sample, rounded for FP16.
- Input padding and both workspaces are initialized to NaN. Stage1 receives
  correctly ordered `(sizes,padded_offsets,block_map)` metadata (Python 90–91).
  Only valid rows are compared, then **both** workspaces' invalid rows must still
  be NaN (93–98), detecting skipped valid writes and unexpected padding writes.
- Stage2 consumes each implementation's own recomputed Stage1 workspace and
  correctly ordered `(sizes,raw_offsets,padded_offsets,block_map)` metadata
  (105–106). Both route dtypes are exercised. Complete outputs must be finite
  and pass the tolerance; the candidate's padding must be exactly zero and the
  last valid token must not be an all-zero row (108–112).
- `describe` separately reports 17-digit max difference, nonzero-difference
  count, actual int16-view bitwise equality, and tolerance failures. The script
  asserts finiteness and tolerance; bitwise equality is **reported**, not required
  for the final PASS. The concluding `all_tested_bitwise` flag preserves that
  distinction.

This script invokes compiled Stage1/Stage2 builders directly. It is a genuine
current-input two-stage chain test, **not a `run_kernel` entry/dispatch test**,
and it does not time execution or use an independent mathematical/OJ reference.
It covers an empty expert block within a positive-token fixture, not the
all-zero-valid or zero-padded host special cases. Those unchanged host paths
must not be claimed as GPU-tested by this particular script.
