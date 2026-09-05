# v746: independent E64 Stage1 extension audit

Scope: local Python/source analysis and, when available, the archived E64
generated Stage1 source. No GPU execution, timing, ISA or hardware profiling is
performed by this audit. Stage2 generated code is outside its scope.

## Python isolation

The complete module AST equals v745 after exactly one executable edit in
`_get_stage1`: `num_experts == 32` becomes `num_experts in (32, 64)`
(v746 Python line 1670). All other 15 functions are source-text identical,
including every builder, `_get_stage2`, `_get_workspace` and `run_kernel`.
Both versions passed 64 getter/cache-return checks spanning E16/32/64/128,
H7168/4096, I2048/1024, padded0/128 and blocks0/1. Only positive
E64/H7168/I2048 changes its selected builder; E32 and other shapes do not.

The actual previous E64 builder is `_moe_stage1_prefetch_giu_merge`, not the
retained `_v527` builder. Its final four statements (two accumulator clears,
guarded GIU/terminal computation, valid-only SwiGLU epilogue) exactly equal the
runtime builder's full-branch AST. Their signatures and pass decorators also
match. Both use k_pack2 and `T.use_swizzle(3 if num_experts == 32 else 2,
order="column")`; E64 therefore retains swizzle2. H7168 still executes steady
K0..110 followed by terminal K111, whose global columns are 7104..7167.

The runtime branch predicate derives only from per-CTA metadata: rows>64 takes
M128, 0<rows<=64 takes M64, and empty rows skip the computation and workspace
writes. The two branches own distinct Gate/Up accumulators. A remains physically
128x64 shared and the M64 branch uses its first64-row view; weight shared remains
128x64. Full/tail T.gemm keywords, GIU order, barriers and final FP16 SwiGLU
conversion are unchanged from the already-audited v745 builder. Actual E64
lowering is independently confirmed below.

Python file SHA256 at this audit:

```text
v745 ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465
v746 9cd17d1b2b8e02fd59fb277d602e9ad03e654b932aa536b43211075dca7e3416
```

## Inherited limits

E64 Stage2 still selects `_moe_stage2_fast_bfrag_prefetch`. The raw-route bounds
clamps, zero-route output-only builder and zero-padded early return are still
E32-only. This candidate does not fix or establish safety of E64 zero-length
route arrays, zero-grid launches or malformed metadata. Source-identical host
behavior and Stage2 are not a new GPU safety proof. Normal positive calls retain
two launches and recompute the current input; only JIT callables and workspace
allocation are cached.

## Actual generated source

Source: `codex_e64_745_746_stage1_codegen.log`, SHA256
`18d72e00e8b139047b8211e4c6289a1e253cb1fcde114bac3f1295fad46a9bb8`.
The following line numbers refer to this log. Complete SOURCE bodies and their
reported hashes/lengths were independently matched:

| E64 source | Characters | SHA256 |
| --- | ---: | --- |
| v745 original GIU | 12945 | `46f83e88168aeffaaa200d7d34d1c957eeec6626448b53c78a756be1ca74363e` |
| v746 runtime M128/M64 | 25102 | `19caa0984f65b59f412cdf9a727853e90d86b7aab7d09ea4dd1acedf04179c8f` |

For each complete generated source, normalize only
`stage1_codegen_review_<number>_kernel` to a common name and replace the older
E32 `rasterization2DColumn<3>` with `<2>`. The E64 baseline then equals the
previously audited E32 v743 original-GIU SOURCE **byte for byte**; E64 v746
equals the previously audited E32 v745 runtime SOURCE **byte for byte**.
Both diffs contain zero lines. The comparison source is
`../v745/codex_e32_743_745_stage1_codegen.log`. Thus there is no additional E64
address, arithmetic, epilogue or synchronization change hidden in this lowering.
Both E64 sources explicitly retain swizzle2 (23 / 245).

### Bounds, fragment layout and output

The candidate's full guard is line249 and its positive-tail guard is435. Both
depend exclusively on CTA metadata. The physical shared allocation is weight
at byte0 and input at byte16384 (236–237): 32 KiB total, not a 24 KiB tail
allocation. Full Gate/Up C are separate float[64] arrays (238–239); tail C are
separate float[32] arrays (241–242). Full A fragments have16 halves, tail A8;
both B fragments have16 halves. `up_prefetch` remains half[32]. These are source
declarations, not physical register allocation.

Captured address expressions were evaluated on CPU for all256 threads and
their vector components, including the final E64 expert63, N tile15, steady
K110 and terminal K111. The following checks passed:

- Gate and Up global expressions (263/271,449/457,509/517) each cover8192
  distinct correct half-elements in their128x64 tile. The final expert/tile
  addresses stay below `64*2048*7168` elements. Gate/Input use uint2 loads;
  current Up prefetch uses uint4. Terminal columns remain7104..7167.
- Full Input copy267 covers128x64; tail Input copies453/513 cover64x64.
  For an E64 edge-fixture final CTA bx71, all addresses remain in the
  72*128-row padded input. Tail copying never touches the upper64 shared rows.
- Every captured tail A LDS site (464/490/524/550) was enumerated over all
  threads, both row tiles, four K16 steps and four scalar components. Its exact
  swizzled offsets are0..4095. All four B LDS sites (467/493/527/553) similarly
  use0..8191. Their expressions equal
  `row*64 + ((col//4 ^ row%16)*4) + col%4`; no A read reaches uninitialized
  shared rows64..127.
- Actual output addresses431/617 cover respectively128x128 and64x128 distinct
  coordinates. At bx71/by15 they stay inside the padded workspace. All full
  valid-row counts65..128 and tail counts1..64 admit exactly
  `actual_rows*128` coordinates. Row-block stride remains bx128 in both paths;
  C slots span0..63 or0..31. Empty rows skip both branches and do not write.

The global/Input/output checks above evaluated90112 scalar addresses. The
eight LDS-site checks were additional. Bounds assume the existing valid block
map and128-padded expert metadata contract, not arbitrary malformed inputs.
The complete-source equality also preserves all448 ascending K16 accumulations
per Gate/Up output and the FP32 SwiGLU/final FP16 conversion; it is not a GPU
bitwise-equality result.

### Actual synchronization

| Scope | Copies to Gate reads | Gate reads to Up overwrite | Up writes to Up reads | Up reads to next tile overwrite |
| --- | --- | --- | --- | --- |
| Full steady | 275 | 293 and294 | 301 | 319 |
| Full terminal | 335 | 353 and354 | 361 | No next tile |
| Tail steady | 461 | 479 and480 | 487 | 505 |
| Tail terminal | 521 | 539 and540 | 547 | No next tile |

All producer/consumer shared-memory transitions are protected in the actual
C++; no barrier is merely assumed from `T.gemm`. Adjacent duplicate barriers
are inherited. There are9 static sites per mutually exclusive branch (18
total); a positive112-tile call executes `111*5+4 = 559` source-level barrier
calls, not twice that number. Empty rows execute neither branch. No next tile
can overwrite the terminal shared data, so no terminal end-K barrier is needed.

The log reports register/spill fields as null: unavailable, not zero. This
source audit does not establish physical register usage, occupancy, spills,
timing or Stage2 safety. No new Stage1 address/layout/synchronization defect
was found in the captured E64 source.
