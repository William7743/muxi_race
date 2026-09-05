# v743 actual Stage2 code-generation audit

## Verdict and scope

No definite address, fragment-layout, synchronization, or branch-initialization error was found in the complete generated FP32-route and FP16-route C++ reviewed here. This is a source-level admission to GPU boundary/correctness testing, **not** a numerical pass, timing result, OJ result, or machine-code race proof.

The codegen fixture is E32, H7168, I2048, padded6144/raw4544, 256 threads, 48 logical M blocks and 56 N blocks. Candidate source at review:

- `probe_v743_v723_e32_stage2_runtime_m64.py`
- SHA256 `67d25409b20cf6417d79375f57edb3770a79fb1a7a619eb8bc3ca9e3b6e0e7ec`.

The independent Python/host/geometry audit is in [audit_v743_cpu.py](audit_v743_cpu.py). This document additionally checks the **actual generated C++**, including compiler changes to the nominally preserved full-M path.

All line numbers below are **source-relative**, with the first `#include` after `SOURCE_BEGIN` numbered 1; they are not raw log line numbers.

## Artifacts and exact source identity

| Artifact | Generated source SHA256 | Characters | Static barrier sites |
| --- | --- | ---: | ---: |
| v723 FP32 in [combined log](codex_e32_723_743_codegen_fp32.log) | `36c433f89d966c69fddcc9d6a015f954c6cb8a7ce5a045c78b8d31a687af5aa7` | 18278 | 3 |
| v743 FP32 in combined log | `d318c28eac64a820d1f40d0e10c369592e778e21b1966ff8f258c11c4d535582` | 32771 | 6 |
| v743 FP16 in [FP16 log](codex_e32_743_codegen_fp16.log) | `b2c789e2c1dd291649593ea7523609866f03320e825715c8e8dacb7ff3be8bfb` | 33340 | 6 |

Raw-log SHA256:

- Combined FP32: `03857b5f05b60bd88797e68a33785809c0a0d80a803b4069e5c690bcb38ad42d`.
- Candidate FP16: `d990e4b6d837324c4d5b562c4da5a8f9d757e4b19e47313a4c306cd50b2a09bb`.

Source extraction honored each metadata `source_characters`, excluding log/printing newlines. All three source hashes were independently recomputed and matched.

## Actual full-path changes versus v723

The new source is not simply the old binary with an extra branch:

1. Up/Down shared-memory physical bases swap. v723 has Up at byte0, Down at byte16384; v743 has Down at byte0, Up at byte16384. Uses follow the corresponding pointer consistently.
2. v743 branches directly on `remaining = group_size + padded_start - bx*128`: full when remaining>64. Under that uniform fact, the compiler removes the repeated `condval=32/0`, `condval_1`, and terminal `condval_4` guards.
3. The full steady loop becomes literal `k<31`, versus `k<condval_1-1`. Both next-tile copies lose the old `if(k<31)`/zero temporary and become direct uint2 copies. Up-next address arithmetic also simplifies from 64-bit casts to 32-bit integer expressions; the largest reviewed Up element offset is 12582911, safely in range.
4. The full epilogue's outer test simplifies `max(0,min(128,remaining))==128` to `min(128,remaining)==128`, valid under remaining>64. The per-row guard, raw-route clamps, multiplication and conversion remain.
5. The original terminal RAW barrier is now inside the full branch; previously it was outside the positive-compute conditional. Each active block still follows the same dependency sequence.

The 16 actual full-path LDS assignment lines are byte-identical to v723 after removing indentation. Full A/B fragment shapes, all eight static full-path MMA sites, accumulator indices and per-C K16 order are unchanged. Prologue is still Up copy, clear C, Down copy. The full output coordinate expressions and route addresses are unchanged.

**Attribution warning:** any later performance difference can include the full-path compiler simplifications and shared-base placement, not only M64 tail work reduction. Python-level full-body identity does not imply identical generated code.

## Global copies and shared layout

For half-element coordinates define the physical shared offset

```text
S(r,c) = r*64 + ((floor(c/4) XOR (r mod 16))*4) + (c mod 4).
```

Each actual uint2 copy transfers four adjacent FP16 elements. With copy-loop index i and thread t:

```text
r = i*16 + floor(t/16)
c = (t mod 16)*4
Up   global = (bx*128+r)*2048 + Kbase+c
Down global = expert*7168*2048 + (by*128+r)*2048 + Kbase+c
shared = S(r,c)
```

Initial copies have Kbase=0. Steady-end copies have Kbase=64*(k+1), k=0..30. There is no K32 load.

| FP32 source line | Copy | Rows written |
| ---: | --- | ---: |
| 36 / 45 | Full initial Up / Down | 128 / 128 |
| 128 / 132 | Full next Up / Down | 128 / 128 |
| 261 / 265 | Tail initial Up / Down | 64 / 128 |
| 348 / 352 | Tail next Up / Down | 64 / 128 |

Read-only CPU replay compiled the actual generated integer expressions (removing only C integer casts and renaming built-in indices), then enumerated every thread, copy iteration and vector element; next copies included every k=0..30. First/last logical blocks and expert/N tiles were exercised. **1,835,008 scalar address checks passed**, including unique complete shared writes, vector alignment, expected global source and allocation bounds.

Each full shared tile covers half-element offsets0..8191. Tail Up covers only0..4095, not an incorrectly compacted stride or the upper64 rows. Down still covers0..8191. No tail copy writes a third shared allocation.

## LDS, fragment slots and accumulation order

Actual LDS uses, for microtile q=0..3 and local element l=0..3:

```text
wm = (t & 127) >> 6; wn = t >> 7
row_A = wm*(64 for full, 32 for tail) + tile*16 + (t & 15)
row_B = wn*64 + tile*16 + (t & 15)
column = q*16 + ((t & 63) >> 4)*4 + l
shared offset = S(row_A or row_B, column)
private slot = tile*4+l
```

- Full A tile=0..3, A slots0..15, C64 FP32 elements/thread.
- Tail A tile=0..1, A slots0..7, C32 FP32 elements/thread.
- Both B tile=0..3; B0/B1 each have slots0..15.
- Full and tail A/B/C have separate declared private arrays. There is no source alias or shared mutable fragment layout between branches.

All **32 actual LDS assignment sites / 114,688 lane-local addresses** passed integer-expression replay. The tail Up maximum is4095 half elements; all other shared LDS maxima are8191. Thus the BufferRegion lowering actually retains the correct original stride64/swizzle and does not read the unused Up upper64.

For each K64 tile the generated sequence is A0/B0/B1, MMA0; A1/B2, MMA1; A2/B3, MMA2; A3, MMA3. B0 is overwritten only after MMA0 consumes it; B1 is overwritten only after MMA1. C's vector offset is `i*4+j` in both branches (full i<4, tail i<2; j<4). All **16 static MMA sites** were checked against this slot sequence. Every C element accumulates K16 offsets0,16,32,48 in order, for steady K0..30 then terminal K31; global reduction coverage is0..2047.

No per-row reduction mixes padded rows with valid rows. Private fragments are cleared/loaded before their first use in their selected branch. Declarations outside the branch do not imply a read of the other branch's uninitialized arrays: all uses remain inside the matching branch.

## Synchronization and branch convergence

| Dependency | v723 FP32 source | v743 full source, both dtypes | v743 tail FP32 / FP16 source |
| --- | ---: | ---: | ---: |
| Producer copies -> first/current LDS (RAW) | 54 | 48 | 268 / 274 |
| Final current MMA/LDS -> next shared overwrite (WAR) | 131 | 125 | 345 / 351 |
| Last next-tile copies -> terminal LDS (RAW) | 156 | 135 | 355 / 361 |

The loop-head RAW barrier handles both the initial tile and the preceding loop iteration's next-tile copies. The steady-end WAR barrier is after the final current-tile MMA and before either shared buffer is overwritten. The separate terminal RAW barrier protects the copies made by k=30. There is no later shared reuse after terminal MMA, so an additional terminal-exit barrier is not needed for these source dependencies.

The full selector is source33 in both dtypes. Tail selection is FP32 source253 / FP16 source259. Both depend only on block-uniform group metadata and logical block x; all 256 threads take the same full/tail/zero route and have the same31-iteration K trip count. No thread-dependent condition surrounds a barrier. Per-row output masking occurs only after all barriers.

Six is the number of **static** source sites, not six executed barriers per block: an active full or tail block executes31*(RAW+WAR)+terminalRAW=63; the zero branch executes none. The branches are mutually exclusive and do not exchange shared data at their merge, so no cross-branch merge barrier is required.

## Raw route loads, output ownership and zeros

FP32 actual scalar route sites are on source215,233,434; FP16 sites are215,236,440. Each line constructs a four-lane local vector from **four scalar global accesses**. All12 accesses independently contain

```text
max(0, min(raw_start + bx*128 - padded_start + output_row, 4543)).
```

There is no global uint2/float4 load from an unclamped route pointer: the vector operations here target local temporary arrays. Full output_row uses wm*64; tail uses wm*32. The per-row guard can follow a route load, but the load is already clamped. 163,840 thread/index evaluations including lower/upper-clamped examples passed.

For valid rows and valid group metadata, the index is the original raw (not padded) route index. For invalid padded rows, even an in-bounds route value belonging to another raw row is discarded by zero output. No route load occurs on remaining<=0.

The actual FP32 output stores are:

| Source line | Store geometry | Unique FP16 elements |
| ---: | --- | ---: |
| 227 / 249 | Alternative full epilogues:128x128 | 16384 each |
| 451 | Tail computed/masked rows0..63 | 8192 |
| 456 | Tail unconditional zero rows64..127 | 8192 |
| 462 | Empty unconditional zero rows0..127 | 16384 |

Computed output uses four-element uint2 stores:

```text
row = wm*(64 for full,32 for tail) + floor(i/4)*16 + (t mod16)
col = wn*64 + (i mod4)*16 + floor((t mod64)/16)*4 + vector_lane
```

Tail upper-zero uses eight-element uint4 stores:
row=64+i*16+floor(t/16), col=(t mod16)*8+vector_lane, i<4.
Empty-zero uses the same formula without64, i<8. Add bx*128 to row and by*128 to col.

Enumeration of each actual store expression proved exact coverage, no duplicates within a selected path, no overlap between lower/upper tail stores, and no uncovered padded output. First/last output tiles remain inside the reviewed6144x7168 output allocation. A partial tail explicitly zeros its invalid rows within0..63 as well as all64..127. Partial full blocks similarly zero rows>=remaining. The empty path contains no Up/Down/route load and reads no uninitialized C.

The candidate's host dispatcher retains v723's separate zero-valid and zero-padded behavior; therefore this positive-route codegen is not called with route length0. General host/metadata boundary conditions are checked by the independent CPU audit, not inferred from the literal4543 in this fixture.

## FP16 comparison and resource limits

A complete FP32->FP16 source diff showed only kernel numbering, route input/local types, route packing, FP16-to-FP32 conversions and resulting temporary numbering. All actual global/shared copy expressions, LDS assignments, raw-route index expressions and output-store addresses match exactly. FP16 route values are converted to FP32 before multiplying FP32 C; output is then rounded to FP16 using the same conversion as the FP32-route path. No FP16 accumulation was introduced.

The generated shared address footprint remains **32768 bytes (32KiB)**: two nonoverlapping128x64 FP16 tiles at offsets0 and16384. Tail saves operations, not the allocation size; it uses only the first8192 bytes of its Up tile. No additional shared region appears.

Both logs report `reported_n_regs=null` and `reported_n_spills=null`: these are unavailable values, not zero. Array lengths and mutually exclusive source branches do not establish physical register allocation, register reuse, absence of spills, or active occupancy. The whole compiled kernel may be constrained by its larger branch or by compiler allocation across both branches. `__launch_bounds__(256,1)` is not a residency measurement.

At the code-generation audit stage, GPU boundary checks and full-entry correctness/timing were still pending. Their subsequently completed results are recorded separately in README.md and PROFILING.md. This code-generation audit alone does not establish performance or OJ acceptance.
