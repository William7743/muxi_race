# v750 E16 Stage2 generated-source audit

Scope: independent read-only review of the complete FP32-route generated source in [codex_e16_750_stage2_codegen_fp32.log](codex_e16_750_stage2_codegen_fp32.log). Compilation completed successfully using the installed `/opt/tilelang-metax-v0.1.10` tree. This audit does not add GPU execution or performance evidence.

## Identity and complete comparison

- Python candidate SHA256: `9fe1ea13e226907071cffe3bdebf7e1c3f6d2f371b7818761c922fb79beeb207`.
- Extracted E16 generated source: **32,783 characters**, SHA256 `1ef1fd9a2acf55a0d404d735f7552fc1d822b8c18ee223096d409803e247e1a8`; independently recomputed, matching the log JSON. Raw log SHA256: `e80ac6fcb55008d240c48d5a85a32ecd910339c15d3dc52bb0c961d57ea2885d`.
- E64 comparison: [v747 FP32 source](../v747_v748/codex_e64_747_stage2_codegen_fp32.log), **32,771 characters**, SHA256 `abfc6d6d1213a19188d0ca31e726b22bd0cb65c52e2f4f9b5a177f265b454e21`. v748 uses the same Stage2 builder/dispatch for this E64 signature; this is not a separate v748 compilation claim.

The complete E64 source becomes **byte-for-byte identical** to E16 after the following line-scoped constant substitutions. Exactly **18 source lines** differ; function names, shared addresses, local declarations, branches, LDS/MMA sequence, and all other text remain unchanged. Global address strides below count FP16 elements; shared offsets count bytes.

| Source sites | E64 → E16 differences |
| --- | --- |
| Four Up global-copy lines | row `2048→8192`; 16-row stride `32768→131072`; M128 CTA stride `262144→1048576` |
| Four Down global-copy lines | same I-strides as Up; expert stride `14680064→16777216` = H×I; existing 64-bit address arithmetic retained |
| Two steady loop headers | `k < 31` / `k_1 < 31` → `< 127` |
| Three route-load lines, four scalar expressions each | upper clamp `9087→2271`, lower clamp remains 0 |
| Five output-store lines | H stride `7168→2048`; 16H `114688→32768`; 32H `229376→65536`; 64H `458752→131072`; 128H `917504→262144` |

The substitution check changes only the RHS global addresses of Up/Down copies, so it cannot accidentally normalize the unchanged tail LDS constant `2048`. There are no extra copy, barrier, or MMA sites hidden by the comparison.

## Dataflow and boundary proof

- Both full and tail branches preload K64 tile 0, execute steady tiles **0–126**, and then consume terminal tile **127**. The last steady fetch uses `k*64 + 64`, so its final scalar is `126*64+64+63 = 8191`. No tile 128 load is emitted.
- Each steady/terminal sequence consumes four A K16 slices using one reused A fragment, with alternating B0/B1 buffers in the order A0/B0, A1/B1, A2/B0, A3/B1. Full A fragment has 16 half values and C has 64 floats per source thread; tail A has 8 and C has 32; both B fragments remain 16 each. These are source-array sizes, not physical register counts.
- Full Up copy covers 128×64; tail covers only the first 64×64 of the same allocation. Enumerating 256 threads, copy-vector lanes, and all four K16 LDS slices confirms each Up LDS address is within its initialized shared view. All Down LDS indices stay within 128×64. LDS address expressions are otherwise exactly the previously reviewed E64 source.
- Shared allocations remain Down at byte 0 and Up at byte 16,384, totaling 32 KiB, with column swizzle 2. Six static `__syncthreads()` sites remain: entry-LDS, pre-overwrite, and terminal-LDS for each mutually exclusive branch. These protect the same read/write transitions as E64; six source sites are not a measured dynamic barrier count.
- For a correctly mapped CTA, full selection is remaining rows >64; otherwise positive rows select M64, and nonpositive rows select all-zero output. Full partial epilogue zeros invalid rows; M64 epilogue zeros invalid rows in its first 64 and separately zeros rows 64–127. Zero branch writes all 128 rows without Up/Down/route tensor loads.
- Independently enumerated scalar addresses for all generated vector stores and actual rows **0–128**: every 128×128 output tile is covered exactly once on its chosen branch; exactly `actual_rows×128` elements are valid and the rest select zero. This is an address/control-flow model, not a numerical device comparison.
- All **12** emitted route expressions use `max(0,min(raw_start + block_start + local_row - padded_start,2271))`. Four values in each `make_float4` repeat the same row weight, not four consecutive route rows. Partial epilogues still hoist the clamped load ahead of their valid-row condition; the clamp bounds these accesses for this nonempty 2,272-route signature. Raw-empty correctness relies on the separate no-load host-selected zero kernel; this generated signature does not prove empty-array safety.

## Limits

This audit covers E16/H2048/I8192 with FP32 routes and the logged nonempty specialization only. It does not certify other route dtypes, malformed group maps, zero-grid device behavior, cross-shape paths, binary instruction equivalence, bitwise numerical equality, or speed. `reported_n_regs` and `reported_n_spills` are null: unavailable, not zero. No inference of occupancy or spills is made. Probe source and README were not changed.
