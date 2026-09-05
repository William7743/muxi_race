# v735: actual generated Stage1 address and synchronization audit

2026-09-05. Scope: local read-only review of
[codex_e32_735_codegen.log](codex_e32_735_codegen.log). No GPU, SSH or Git
operation was performed by this audit. All line references below refer to
that log, not the Python probe.

## Identity and limits

- Frozen tested Python source:
  `probe_v735_v731_e32_stage1_interleaved_gu32_warp2x2.py`,
  SHA256 `7ea5bfcfb07edaa91741faf2644596b7fc636ce5e0d8ec6c8068ad6279995136`.
- Actual generated C++ SHA256:
  `8e64f732cc302616c3f5ae686ce07c8fa5d8ebfff972452015e20ba6fbc78a53`,
  10638 characters. Removing only print-added trailing newlines from the
  SOURCE_BEGIN/SOURCE_END capture reproduces the reported hash.
- Raw log SHA256:
  `3b976e55898362a08c6f26a7c0532a43d3d3140beb80367e82181541fd085a61`.
- E32/H7168/I2048, M128/totalN128/output64/K64, 256 threads, k_pack=1,
  2x2 warps with warp64x64. Compilation succeeds.
- The source/CPU isolation audit is separately in `audit_v735_cpu.py`.
  This document checks the **actual generated copy/load/store expressions**,
  not merely the intended Python geometry.

Neither audit establishes device numerical correctness, performance, physical
register usage, spills or active occupancy. Those require separate measurement.

## Four B segments and actual copy addresses

Bshared starts at byte offset 0, Ashared at 16384 (lines 17-18). Both contain
8192 FP16 elements; their nonoverlapping allocations total 32768 bytes.

| Current-K copy | Physical B rows | Half-element offset | Global N within output64 | Steady line | Terminal line |
| --- | --- | ---: | --- | ---: | ---: |
| Gate0 | 0..31 | 0 | 0..31 | 35 | 79 |
| Gate1 | 64..95 | 4096 | 32..63 | 39 | 83 |
| Input A | separate A tile | 0 | 128 token rows | 43 | 87 |
| Up0 | 32..63 | 2048 | 0..31 | 47 | 91 |
| Up1 | 96..127 | 6144 | 32..63 | 51 | 95 |

Thus the **physical** B row order is Gate0, Up0, Gate1, Up1, while the
synchronous copy issue order is Gate0, Gate1, Input, Up0, Up1. There is no
asynchronous copy or precomputed global concatenation.

All copies are `uint2`, eight bytes/four half values. Each B copy loops
twice per thread; the A copy loops eight times. With 256 threads, each B
segment receives 2048 half elements and the A tile receives 8192, exactly
once. The actual K64 swizzle of logical coordinate (r,c) is

```text
offset = r*64 + (((c//4) XOR (r%16))*4) + c%4
```

Offsets 0/2048/4096/6144 shift by complete 32-row segments. Their multiples
of 32 do not alter r%16, so the copy swizzle and subsequent full-B read
swizzle agree across all four boundaries.

The terminal Gate1/Up1 global offset 236480 is 32*7168 + 7104:
the second 32-column weight half at terminal K111, not a different K tile.

## Independent replay of captured expressions

A local Python check extracted all ten captured `uint2` copy assignments,
four LDS load assignments and the final output-store index by regular
expression. It removed only integer casts, renamed thread/block variables,
compiled arithmetic/bitwise expressions with no calls, attributes or array
subscripts allowed, and evaluated the original addresses as integers.

The replay covered expert 0/31, bx 0/47, by 0/31, and K0/K110/terminal K111:
24 boundary combinations. Within each it enumerated every thread, copy-loop
iteration, four-half vector lane, ki=0..3, operand tile=0..3 and local=0..3.
All checks passed:

- Exactly 8192 in-range, unique writes to each shared allocation.
- Every LDS address is initialized by the current K's appropriate global
  input, Gate or Up element.
- All A and all B shared half-elements are read exactly twice, reflecting
  two column warps for A and two row warps for B.
- Each operand local slot is tile*4+local, within 0..15.
- No stale previous-K or wrong-half value is required by the generated loads.

The LDS lines are 57/62 for steady and 103/108 for terminal. They use

```text
warp_m = (threadIdx.x // 64) % 2
warp_n = threadIdx.x // 128
A row  = warp_m*64 + tile*16 + lane%16
B row  = warp_n*64 + tile*16 + lane%16
K col  = ki*16 + (lane//16)*4 + local
```

These agree with the default official emitter's thread extraction. B physical
row p maps to Gate if p%64<32, otherwise Up, with output-column index
(p//64)*32+p%32. The full captured expressions, not only these simplified
formulas, were used for the replay.

## Shared-memory barriers and empty blocks

| Dependency | Generated barrier | Relevant surrounding operations |
| --- | ---: | --- |
| Steady producers before any A/B reads | 53 | all five copies 35..51, then reads 57/62 |
| All steady reads before next-K overwrite | 75 | after the complete ki/load/MMA loop |
| Terminal producers before terminal reads | 98 | copies 79..95, then reads 103/108 |

The positive-row guard at 31 is block-uniform. It contains the steady loop,
end-K barrier and terminal copies. Barrier 98 is outside the guard, followed
by the same positive-row guard at 99 around terminal reads/MMA. Therefore
empty expert blocks still reach that final barrier but perform no undefined
operand read or output write. There is no divergent CTA barrier caused by
individual row validity.

There are three static sites, not three dynamic barriers per active CTA:
positive blocks execute 2*111+1=223 barriers; empty blocks execute the single
unguarded terminal barrier. The last steady end-K barrier also protects the
following terminal writes. No additional barrier is needed after terminal
MMA because shared memory is not reused before kernel exit.

All required producer/read and read/overwrite dependencies are present in this
capture; this does not assume that replacing T.gemm with an emitter always
gives automatic synchronization.

## C slots, output pairing and mathematical order

The source declares `float gu_local[64]`, `half_t input_matrix[16]` and
`half_t weight_matrix[16]` (19-21). C initialization writes all 64 floats.
The MMA vector offset at 68/114 is row_tile*4+col_tile; with four floats
per vector it covers local slot row_tile*16+col_tile*4+local, exactly 0..63.

The current official 64-lane output map remains
row16=lane%16, col16=local+(lane//16)*4. Generated epilogue lines 122-126
are equivalent to

```text
row = warp_m*64 + row_tile*16 + row16      # row_tile=0..3
col = warp_n*32 + pair*16 + col16         # pair=0..1
gate_slot = row_tile*16 + pair*4 + local
up_slot = gate_slot + 8
out[(bx*128+row)*2048 + by*64+col]
```

Within each warp, Gate uses C column tiles 0/1 and Up uses 2/3. Thus +8
local floats pairs the same global Gate/Up column despite the physical
B interleaving. This is deliberately different from v731's +16 mapping.

The actual output index and all three C-index occurrences at line 126 were
also replayed. For bx0/47 and by0/31 they produce 8192 unique stores per full
block with correct global addresses and Gate/Up slots; every row-validity
threshold 0..128 selects exactly actual_rows*64 outputs. Across by0..31 the
64-column tiles cover I2048 once. The unclamped remaining-row comparison in
generated code is equivalent to the Python max/min guard because candidate
rows are already confined to 0..127.

Each output accumulator sees K0..110 followed by terminal K111, and ki0..3
within each. This is the same 448 K16 chunks at offsets 0..7152 without gaps
or duplicates. SwiGLU is still Up * (Gate * (1/(1+exp2((-Gate)*scale)))) in
FP32 before the FP16 store. Route weighting remains in unchanged Stage2.

## Compiler observations, not performance conclusions

Copy loops and C clearing carry unroll pragmas. Small K16/LDS/MMA and
lane/local epilogue loops remain without explicit unroll pragmas in this C++;
scalar LDS and scalar FP16 output accesses are visible. This is not proof
that native compilation retains each loop, nor that it causes slow execution.

The metadata reports `reported_n_regs=null` and `reported_n_spills=null`.
These mean unavailable, not zero. Local array declarations and the source's
twofold operand-read multiplicity are not physical register counts or measured
shared-memory bandwidth.

**Conclusion:** no source/address, same-column Gate/Up pairing, output-boundary
or CTA synchronization defect was found. This capture is ready for separate
GPU numerical validation; it does not by itself promote v735 for OJ.
