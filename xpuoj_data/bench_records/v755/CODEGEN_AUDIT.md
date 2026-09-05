# v755 Stage1 generated-source audit

Scope: local, CPU-only inspection of `codex_v755_e32_edges_codegen_profile.log`; no additional GPU run, performance conclusion, or OJ claim. Line references below refer to this raw log. Audited E32 / H7168 / I2048, raw2373 / padded4608 / 36 CTAs. Stage1 uses FP16 data independently of the subsequent route-weight dtype.

## Identity and unchanged paths

- Raw log SHA256: `5b6e0ac1b079179d394d81b00ef29520e35e6f1887dc3254ce3e27149809f79b`.
- Tested v755 Python SHA256 recorded in the log: `0b19e84d1695a16bca424ad7fd91f3a51b8baeb4f9e8b3cbf2fc3501224f94de`.
- Recomputed generated-source hashes including the final LF match: baseline v748, 25098 characters, `da1da6e6b8e5146031d5643fee488550f33a181db544c36ed2530adf89b9e17d`; candidate v755, 36867 characters, `2d422a2402982639070c19e28523872ca4c1b2996697ddb473439a669e7f7e16`.
- CPU comparison of complete balanced-brace blocks: M128 at baseline line42 versus candidate447 is identical after trimming indentation only. M64 at baseline228 versus candidate633 is identical after indentation trimming and changing only its outer guard from `0 < remaining_rows` to `32 < remaining_rows`.
- Python AST comparison preserves every pre-existing function except `_get_stage1`; the only new function is `_moe_stage1_e32_runtime_m32_m64_giu_merge`. The new selector is limited to positive-padded, positive-block E32/H7168/I2048. E64, E16, Stage2, `run_kernel`, and cache-key construction retain their prior code. This does not establish identical physical resource allocation for the enlarged Stage1 kernel.

## M32 address and value checks

- Python builder specifies `T.gemm(..., transpose_B=True, policy=T.GemmWarpPolicy.Square, k_pack=2)` on A[0:32,0:64], B128x64, and independent Gate/Up C32x128 fragments. Generated launch remains 256 threads (429). Shared offsets are B=0, A=16384 (431–432), with their existing 16KiB allocations; no reduction of the declared shared footprint is claimed.
- Generated A copies at 837 and 889 each cover exactly shared half indices 0..2047 once. CPU enumeration of all four A LDS expressions (847/869/899/921), 256 threads, four K16 steps and four scalar lanes confirms exactly 0..2047. All four B LDS expressions (849/871/901/923) cover 0..8191. Thus the M32 GEMMs do not consume unwritten A rows32..127.
- Actual global-to-shared expressions at 833/837 and 885/889 were replayed for all threads/vector elements, at the final expert31 / output tile15 / padded CTA35, steady K0 and K110, and terminal K111. They match the shared swizzle of the corresponding current input/weight coordinates; terminal columns are7104..7167, not an extra K112 tile.
- Current-Up register prefetch (841/893, four uint4 loads into 32 half elements per thread) and subsequent shared stores (863/915, eight uint2 stores) were replayed element-by-element. They preserve the same current K tile and swizzled B coordinate. Up is prefetched before Gate GEMM, but shared B is not overwritten until Gate finishes.
- Generated Gate and Up C each declare16 float elements (438–439). For thread `t`, epilogue iteration `i=0..3`, vector lane `v=0..3`, the store at985 maps to row `((t&127)>>6)*16+(t&15)`, column `(t>>7)*64+i*16+((t&63)>>4)*4+v`, local slot `4*i+v`. CPU enumeration proves a bijection over32x128 outputs / 256x16 C slots. The generated MMA calls use those four float4 slots and the corresponding A/B fragments.
- M32 valid-row guard at933 encloses the entire SwiGLU computation and store, and its row is independent of `i` and `v`. Enumeration for every valid-row count0..32 gives exactly `valid_rows*128` stores. Padding inputs may be NaN, but no cross-row reduction is introduced and invalid workspace rows remain untouched. Empty blocks skip all three computation branches; Stage2 retains responsibility for output padding zeros.
- SwiGLU value operations at baseline175–222 and candidate936–983 are identical after renaming only compiler temporaries and Gate/Up fragment names. FP32 arithmetic order and final FP16 round-to-nearest conversion are preserved.

## K order and synchronization

The mutually exclusive uniform branches are remaining rows >64, >32, and >0 (447/633/819). M32 steady loop830 runs K0..110; terminal885–931 processes K111. Each Gate and Up accumulator receives four increasing K16 microsteps per K64 tile (846/868 and898/920): 448 microsteps per output, with no missing or repeated K tile.

M32 steady barriers at845,859,860,867,881 protect respectively A/Gate production, completion of Gate reads before B overwrite (two consecutive source barriers retained), Up production, and completion of Up/A reads before the next K writes. Terminal sites897,911,912,919 preserve the analogous dependencies. There is no final Up-to-next-tile overwrite, so no new terminal end barrier is required. M128, M64 and M32 each contain nine static barrier sites; a positive CTA executes only its selected branch (111*5+4=559 source-level barrier calls), not all27 sites. These are generated-source counts, not disassembled hardware-instruction or measured stall counts.

## Existing numerical evidence and limits

The raw log reports six actual `bitwise_equal=True` comparisons: two fresh-input Stage1 checks and two route dtypes (FP32/FP16) for each of two chains (995/1001/1007/1009/1011/1013). All also report finite valid results, maximum difference0 and zero tolerance failures. Stage1 padding stayed NaN (996/1010); all four chain padding checks report zero output. This fixture includes empty, tiny, M64 and M128 blocks, boundaries31/32/33 and63/64/65, and a final one-row block. These are comparisons against v748 for the recorded seeds and fixed weights, not an independent mathematical reference, exhaustive input proof, `run_kernel` performance test, or OJ validation.

No structural address, accumulation-order, padding-store, or synchronization defect was found in this bounded audit. Source local-array declarations do not establish physical registers, spills, occupancy, or performance. Performance and final tested/final-file identities belong to the separate run records. No further exploration was started.
