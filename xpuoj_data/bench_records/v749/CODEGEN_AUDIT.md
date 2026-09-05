# v749: independent E16 Stage1 generated-source audit

Scope: local source/codegen review, no GPU execution or timing. Line numbers
refer to `codex_e16_745_749_stage1_codegen.log` (SHA256
`e62f1f947d407292b24e6d32cc3f300e434eb927aaf8c0f06d8426be57a155b0`).
The complete captured SOURCE hashes/lengths were independently verified:

| Source | Characters | SHA256 |
| --- | ---: | --- |
| v745 E16 baseline | 9900 | `b08483b59b2beab776a580561b4f12b6de5565f046e5fc1120cea3c49655c1c6` |
| v749 E16 runtime | 17056 | `2422ff403ab2f8538b156085c16608455e580155ce560bcff4678d9b115dc83d` |

Python v749 at review: `1f057c8ee74f1385cb445a2bb8b9a3c89f6503522feba8e93d5a82c0ff853270`.
The actual donor is v745 `_moe_stage1_prefetch`, not its GIU/terminal builder.
An independent AST comparison confirms identical builder signature/decorators,
the complete full-branch body, and the existing
`active_k_steps = T.if_then_else(actual_rows > 0, T.ceildiv(hidden, bh1), 0)`.

## Retained computation and compiler simplification

Full/tail CTA-uniform guards are210/338. Both positive branches preserve
Input -> Gate -> current-Up prefetch -> Gate GEMM -> Up shared copy -> Up GEMM.
All32 K64 tiles remain in the ordinary loop, including its last end-K barrier;
there is no GIU reordering or separate terminal K. Each output accumulates128
ascending K16 steps. Full generated GEMM/shared-transfer/barrier block234–280
equals baseline75–121 after trimming indentation. Baseline/full/tail SwiGLU
operation and final FP16 conversion sequences (126–174 / 285–333 / 413–461)
are equal after consistent temporary/C renaming, not a GPU bitwise claim.

There is a meaningful compiler control-flow difference: baseline35–41 computes
`condval=32 or0`, and its three global-load groups retain `k<32` guards with
zero fallbacks (46/57/68). In v749 the enclosing positive branch allows the
compiler to infer `k<32` directly (221/349), removing those redundant load
guards. The Python loop bound was **not manually changed**. This source change
alone does not quantify performance or isolate the benefit of M64 geometry.

## Shared/fragment layout, addresses and synchronization

Both sources allocate Input at byte0 and weight at byte16384 (candidate197–198):
32 KiB shared remains allocated in either branch. Swizzle2 remains at206.
Full Gate/Up C are separate float[64]; tail C are separate float[32] (199–203).
Full A arrays have16 halves and tail A8; all B arrays have16; current-Up
prefetch remains half[32]. No physical register/spill inference follows.

CPU enumeration evaluated245760 scalar addresses from the actual expressions:

- Global copies224/228/232 and352/356/360, for all256 threads/components at
  K0/K31, expert15, N tile63 and bx18, match the expected current-input tiles.
  Full Input covers128x64, tail64x64; Gate/Up both128x64. Addresses stay within
  E16/H2048/I8192 and the2432-row padded edge fixture.
- Full A LDS239/265 spans shared half offsets0..8191; tail367/393 only0..4095.
  All B LDS242/268/370/396 spans0..8191. Every site matches the swizzle
  `row*64 + ((col//4 ^ row%16)*4) + col%4`. No tail A read touches the
  uninitialized upper64 rows.
- Current-Up fragment-to-shared stores258/386 cover8192 unique half offsets
  and preserve each thread's loaded `(row,K-column)` label. There is no
  prior-K/prior-call fragment reuse.
- Output334/462 covers128x128 or64x128 distinct coordinates with unchanged
  external bx128 and N tile stride128. All applicable row predicates select
  exactly `actual_rows*128` writes; no padding writes, and empty blocks skip
  both computation branches. Valid metadata/padding is assumed, not arbitrary
  malformed inputs.

| Branch | Copies -> Gate reads | Gate reads -> Up overwrite | Up writes -> Up reads | Up reads -> next Input/Gate overwrite |
| --- | --- | --- | --- | --- |
| Full | 236 | 254 and255 | 262 | 280 |
| Tail | 364 | 382 and383 | 390 | 408 |

All producer/consumer transitions are protected in actual generated C++;
automatic barriers were inspected, not merely presumed from `T.gemm`. The
adjacent duplicate barriers are inherited. Five static sites per mutually
exclusive branch (ten total) mean160 source-level barrier calls per positive
32-tile CTA, not320. Empty rows execute neither branch. The log reports
register/spill fields as null, meaning unavailable rather than zero.

No new address, current-K, fragment-layout or synchronization defect was found
in this captured E16 Stage1. Stage2, whole-entry behavior, GPU precision and
performance require their separate tests.

## Independent E16 edge-helper review

`remote_v749_v751_e16_edges.py` metadata was executed without torch/TileLang:
raw1213/padded2432/19 CTAs,1219 padding rows,1 empty/10 short/8 full blocks.
The final raw token1212 maps to padded row2304, correctly `out[-128]`.
Dimensions, NaN initialization, fixed matrix seeds75110/11/12 and fresh x/route
seeds75101/02 are consistent. Both route dtypes reuse their underlying sample.
The isolated Stage2 uses baseline workspace; the combined chain uses the
candidate's own recomputed workspace. The explicit clamped-M128 Stage2 reference
is deliberately not the old v745 E16 dispatcher. Tolerance and actual int16-view
bitwise comparisons are separately reported; only tolerance is required by PASS.

Positive-input checks directly invoke builders, not `run_kernel`. Empty-route
checks instrument the Stage1 getter; zero-padded checks also instrument the
workspace getter, but do not independently count Stage2 calls/device launches.
They require the planned v751 composition's E16 empty guards, not Stage1-only
v749. This is a local baseline-comparison diagnostic, not an independent
mathematical/OJ oracle or performance test.
