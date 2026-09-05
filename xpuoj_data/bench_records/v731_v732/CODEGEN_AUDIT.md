# v731 / v732 attempt 2: generated Stage1 audit

2026-09-05. Scope: the captured E32/H7168/I2048 Stage1 C++ in
`codex_e32_731_732_codegen_attempt2.log`; no GPU execution or probe edits by
this auditor. This is an address/synchronization audit, not a performance result.

## Captured versions

| Version | Python SHA-256 | Generated-source SHA-256 | Source lines in log |
| --- | --- | --- | --- |
| v731 | `f1271c866444c1b9921b721f47c4b6f7ba16b95e33f377669d210578296f4f5c` | `4132f97e6420a6fdd28e73ef094c0067a135b4c024903816bc681064cbe4a888` | 6-117 |
| v732 | `3bd4f4ce6c2216be2b69e334b76d9411820053e28668ce7c22f42c265cd14797` | `d7d0ee41065bd84376003ee43fd6a9816a8ce2f9ba48a4c2b6ba7f5818da32b9` | 123-234 |

Source lengths are 8240/8299 characters. Removing only the extra newline added
by printing between `SOURCE_BEGIN` and `SOURCE_END` reproduces both metadata
hashes. Both compile successfully. The former high-level paired-fragment
epilogue was replaced before this capture; attempt 1 is not the audited code.

## Shared addresses and synchronization

| Property | v731, K64 | v732, K32 |
| --- | --- | --- |
| Bshared / Ashared offsets, bytes | 0 / 16384, lines 17-18 | 0 / 8192, lines 134-135 |
| Inferred total of two shared tiles | 32768 bytes | 16384 bytes |
| Steady Gate / A / Up copies | 35 / 39 / 43 | 152 / 156 / 160 |
| All shared writes before first LDS reads | barrier 45, reads 49/54 | barrier 162, reads 166/171 |
| End-K read-before-overwrite protection | barrier 67 | barrier 184 |
| Terminal Gate / A / Up copies | 71 / 75 / 79 | 188 / 192 / 196 |
| Terminal write-before-read protection | barrier 82, reads 87/92 | barrier 199, reads 204/209 |

The first two barriers occur within a block-uniform positive-row guard. The
terminal barrier is outside the guard, followed by the same guard around the
terminal LDS/MMA operations. Therefore an empty expert block still reaches that
barrier but does not read uninitialized shared operands or write Stage1 output.
No barrier is needed after terminal MMA: shared memory is no longer consumed or
overwritten by this kernel. No shared overwrite occurs inside the K16 loop.

There are three **static sites**, not three barriers per active CTA. For a
positive block the visible source executes 223 barriers for K64
(`2*111+1`) or 447 for K32 (`2*223+1`). All required inter-wave shared dependencies
are protected in this captured source; this does not assume T.gemm inserts a
barrier when the direct emitter is used.

Every global-to-shared copy is `uint2`: eight bytes/four FP16 values. Gate fills
the first 64 B rows; Up fills the second 64 at offsets 4096/2048 FP16 elements.
Each A/B tile is filled exactly once, and the two shared allocations do not
overlap. For logical row `r`, column `c`, the generated layouts reduce to:

- K64: `r*64 + (((c//4) ^ (r%16))*4) + c%4`.
- K32: `r*32 + (((c//4) ^ ((r//2)%8))*4) + c%4`.

Both are permutations within the row, and the generated LDS addresses use the
same layout as their producers. K32 is **not** the K64 XOR mask blindly applied
to a narrower allocation.

Independent CPU evaluation used the actual captured C++ address expressions
(integer casts removed, no floating-point addresses): all 256 threads, first
and last steady iterations, and terminal; highest captured expert=31, bx=47,
by=31. Every four-half global-to-shared copy expansion had a unique in-range shared
destination. Every subsequent scalar LDS read matched its expected current-input
or Gate/Up global element, including the boundary between the B halves. All
checks passed for both versions. Remaining steady iterations use the same
affine K address term. No current-input routing or result cache was introduced.

## C slots, output coordinates, and mathematics

Both sources retain `float gu_local[64]`, `half_t input_matrix[8]`, and
`half_t weight_matrix[32]` (19-21 / 136-138). Official emitter `mma` accepts the
local C buffer: it computes fragment-dependent strides only for A/B, then uses
`C_local_buf.data` with vector offset `i*8+j` (downloaded
`mma_macro_generator.py`, lines 395-432). C vector indices 0-15 cover exactly
64 float slots, initialized at 27-29 / 144-146.

The current official `utils.py:4-5` resolves the store map to
`mma_layout.py:101-104`: `row=lane%16`, `col=local_id+(lane//16)*4`.
With four row-warps and one column-warp, the generated epilogue (106-113 /
223-230) is consequently:

```text
row = warp_m*32 + row_tile*16 + lane%16
col = col_tile*16 + (lane//16)*4 + local_id
gate_slot = row_tile*32 + col_tile*4 + local_id
up_slot = gate_slot + 16
```

Here row_tile=0..1, col_tile=0..3, local_id=0..3. Gate slots are 0-15 and
32-47; Up slots are 16-31 and 48-63. They are paired on the same thread and
match MMA column tiles j and j+4. These are not two contiguous 32-slot halves.

The final generated store at 110 / 227 is exactly
`up_logits[(bx*128+row)*2048 + by*64+col]`. CPU enumeration of the captured
offset gives 8192 unique in-range stores per full block; the official-map
enumeration covers all 16384 C slots and all 8192 pairs. For every
`actual_rows` from 0 through 128, the row guard selects exactly the valid rows;
by=0..31 covers all 2048 output columns once. Padding workspace is left
unwritten as in the base Stage1, and Stage2 is unchanged.

Each output accumulator sees K16 chunks in increasing order. K64 uses
111 steady tiles then terminal offset 7104; K32 uses 223 steady tiles then
terminal offset 7136. Both cover the same 448 K16 chunks, offsets 0..7152,
without omission or duplication. The resulting SwiGLU expression is still
`Up * (Gate * (1 / (1 + exp2((-Gate)*scale))))`, in FP32 before one FP16 output
conversion. No route-weight multiplication was moved into Stage1.

## Retained compiler behavior and limits

- Copy loops and C initialization carry `#pragma unroll`.
- K16, A/B scalar LDS, MMA tile, and output loops have no explicit unroll pragma
  in this generated C++ (46-65 / 84-104 / 106-113; 163-182 / 201-221 / 223-230).
- LDS loads and FP16 output stores are scalar in this source, unlike the
  vectorized epilogue visible in earlier baseline captures. This is a concrete
  code-generation difference worth separating in later experiments, **not**
  proof that the native compiler leaves the loops rolled or that loop overhead
  caused a measured regression.
- `reported_n_regs` and `reported_n_spills` are null. Local array declarations
  are not physical register counts and do not prove presence or absence of
  spills, occupancy, or native instruction scheduling.

Conclusion: no address, C-pairing, or shared-synchronization defect was found in
these two captured Stage1 implementations. GPU numerical results, performance,
and OJ suitability must be recorded separately; this audit alone does not
promote either candidate.
