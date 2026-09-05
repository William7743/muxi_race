# v736–v738 independent Stage2 audit

Scope: local CPU source/AST audit and the archived E32, H7168, I2048,
route-weight **float32** generated source in
`codex_e32_720_736_737_738_codegen.log`. No GPU execution, ISA inspection,
performance measurement or OJ result is implied by this audit.
Line numbers below refer to that full log, not standalone generated C++ files.

## CPU source, dispatch and dependency checks

Reproduce from the repository root:

```text
py -3.12 xpuoj_data/bench_records/v736_v738/audit_v736_v738_cpu.py
ruff check xpuoj_data/bench_records/v736_v738/audit_v736_v738_cpu.py
```

Both passed. The audit imports neither installed torch nor TileLang.

- Every original v720 function except `_get_stage2` is exact source-text equal.
  An independent whole-module AST reconstruction permits only one cloned builder,
  its declared tail transformation, and the new E32/H7168/I2048 selector.
  Stage1, all other shapes, passes, geometry, raw/padded indexing, workspace and
  callable-cache logic are unchanged.
- v736 moves the existing steady end barrier from after MMA3 to after A3 / before
  MMA3. v737 additionally splits the next Up global-to-shared copy through
  `next_up`; v738 instead splits the next Down copy through `next_down`.
  Both new global-to-fragment copies precede the current MMA3; their shared writes
  follow it. The entire source terminal path and epilogue remain unchanged.
- Tagged execution of the actual AST passes K=1,2,3,32,64. Each accumulator sees
  exactly `(k,0),(k,1),(k,2),(k,3)` in order for every k. The A/B tags agree at
  every MMA; the retained A3/B3 tags survive the short prefetch. Global tile reads
  are exactly 0 through K−1 once per input, never K. K=1 has no steady prefetch.
  All shared-overwrite WAR dependencies are protected by the moved explicit
  barrier. This CPU model does **not** invent implicit producer-to-consumer RAW
  barriers; their actual locations are checked below.
- Full-module import with stub JIT builders confirms two fresh calls forward the
  respective current tensors, with exactly two launch requests per call, two
  compiled callables and one reusable workspace allocation. E1/8/16/32/64,
  off-target E32 shapes, and both float16/float32 route dtypes pass. No result
  reuse or correctness/benchmark branch is introduced.
- The actual source epilogue is executed for every `actual_rows=0..128`, both
  route dtypes, with poisoned invalid accumulator/route accesses. Valid rows
  preserve the FP32 multiply and FP16 output bytes; invalid rows write zero.
  The one-valid-row case leaves 127 padding rows after the final raw token.
  Host cases `(padded,valid,blocks)=(0,0,0),(256,0,2),(256,1,2),(512,129,4)`
  preserve baseline behavior. In particular, padded=0 still requests two
  zero-grid launches; these probes do not implement a new empty-input return.

Probe SHA256 at audit time (headers may subsequently be updated by the owner):

```text
v720 2d5605e80220dcecf0e1ae1d86f2edbbb9b60ad2438d2d783f77efe82bb0e774
v736 dd9e3da2282f880dceb2ac6c3807a800202b17ab36b276d38f0e3d6bda838281
v737 193ece7ff8fc7a33821e59f3dc8640217348e15154647dad085ed9e3014828d9
v738 01c3f75016b97d33a7296f3fca68538e7256a0f22df62e5b08645426e85cf5a4
```

## Generated synchronization and fragment lifetime

| Version | Generated source log lines | Actual synchronization sites | Positive K32 dynamic barriers |
| --- | --- | --- | --- |
| v720 | 4–286 | 57 loop head; 134 after MMA3; 159 terminal | 63 |
| v736 | 292–574 | 345 loop head; 413 after A3 / before MMA3; 447 terminal | 63 |
| v737 | 580–868 | 634 loop head; 702 after A3; **723 added after MMA3**; 741 terminal | 94 |
| v738 | 874–1171 | **922 prologue**; 942 after loop A0; 1005 after A3; **1037 after next Up write**; 1049 before terminal B0 | 95 |

These are 31 steady iterations plus one terminal tile. Static counts 3/3/4/5
are not per-iteration counts. Empty-row CTAs execute one outer barrier in
v720/v736/v737 and two in v738; all guards depend only on CTA-uniform metadata.

**v736 is an exact generated-source isolation.** After normalizing only
`stage2_codegen_review_N_kernel`, the entire v720 source becomes byte-equal to
v736 by moving one `__syncthreads()` from immediately after the final steady MMA
loop to immediately before it. There is no automatically reinserted late
barrier. Generated terminal, epilogue, addresses, arrays and vector widths are
otherwise byte-equal, not merely mathematically equivalent.

v737 keeps the requested schedule: A3 loads 697–701; barrier 702; next Up
global-to-private loads 704–713; current MMA3 714–722; extra compiler barrier
723; private-to-Up-shared stores 725–727; original next Down copy 729–739.
Neither the global prefetch nor that extra barrier overwrites the current
`up_matrix[16]` / `down_matrix1[16]`. Loop-head barrier 634 protects both shared
producers before their next readers; terminal barrier 741 does the same.
The whole terminal and epilogue are equal to v720 after consistent loop-variable
renaming (the additional prefetch/store loop shifts `i_N` names).

v738 keeps next Down global-to-private loads 1007–1016 before current MMA3
1017–1025. Original next Up copy 1027–1036 precedes added barrier 1037; Down
private-to-shared stores follow at 1039–1041. On the next iteration A0 can be read
before barrier 942 because the preceding Up stores were already synchronized
at 1037. B0/B1 wait for barrier 942, which protects the later Down stores.
The first iteration instead gets its initial shared data through prologue
barrier 922. Likewise the compiler has moved **terminal A0** to 1043–1047 before
barrier 1049: that A0 is protected by the last 1037 (or 922 for a one-tile path),
while terminal B0/B1 remain after 1049 at 1057–1065. This is a real generated
control-flow change despite the terminal Python AST being unchanged; it is not
an unprotected read. No shared access occurs in the retained final MMA itself.

Across all four generated sources, the ordered 16 shared-to-private assignment
sites and eight MMA statement sites (four steady, four terminal) are exact after
loop-variable renaming, including A/B address expressions, destination C slots,
operand selection and accumulator order. The entire full/partial epilogue is
also exact after consistent `i_N` renaming for v737/v738. No new MMA/layout
aliasing discrepancy was found in the captured C++.

## Actual copy layout and resource declarations

Both shared regions remain disjoint 16 KiB buffers, offsets 0 and 16384,
for 32 KiB total. The existing declarations remain `out_local[64]` floats,
`up_matrix[16]` halves and two `down_matrix[16]` half arrays. v737 adds
`next_up[32]` halves (597); v738 adds `next_down[32]` halves (891).
These C++ arrays are not a physical-register or spill measurement;
`reported_n_regs` and `reported_n_spills` are null, not zero.

The prefetched global read is really `uint4` (16 bytes / 8 halves) at 708 or
1011, whereas the baseline direct global copies use `uint2` (8 bytes / 4
halves). Its private-to-shared write remains `uint2` at 726 or 1040. No async
copy is present. This is both a schedule and a load-width/layout change, not
only a latency-hiding experiment.

The actual captured global address and shared destination expressions were
evaluated for all 256 threads and all 32 private half slots, at bx47, by55,
expert31, steady k30 (the last prefetched tile). Each produced exactly 8192
unique shared half addresses with the correct global value. For thread t and
slot s, the recovered logical coordinates are
`row=32*(s//8)+t//8`, `col=8*(t%8)+s%8`, and the shared half offset is
`row*64 + ((col//4 ^ row%16)*4) + col%4`.
Thus the different 8-half global and 4-half shared partitions agree, including
the final K range 1984–2047. Current MMA fragments are separate arrays and are
not overwritten by either `next_*` buffer in generated C++.

## Important inherited limit: route-load bounds are not fixed

Source-level empty/tail checks pass, but the captured **float32** generated
partial-row epilogue still loads raw route weights **before** its row-validity
condition, without clamping. The baseline proof is load 264 before guard 265;
the corresponding candidate loads are 552, 846 and 1149. The whole epilogue
equivalence above confirms the same risk is inherited by all three probes.
For a final expert tail, padding lanes may read past the raw route-array end;
an empty raw array cannot be made safe by its later zero-output branch.
These are not v723 memory-safety probes and must not be described as repairing
that issue. This log does not establish generated float16-route safety either.

Generated source hashes, recomputed from the full SOURCE_BEGIN/END bodies and
verified against all four metadata records:

```text
v720 19158 chars ea5b117bf75137e597065e618110f15f361f5bc9373ff17fb5ba032a85946214
v736 19158 chars 78ab2517a0807f3f5d9aff194832cd990ab5edbed06d57e73e651087da35f523
v737 19482 chars 0ce0e2514feaa6de541a0497124f36109b4894b32e78f0cfc87c7a54b915951d
v738 19678 chars c6b9bd5a9e44c0df368dcb832d4f7c3208c9fb05176016bf2d42115fc45ffde1
```

Conclusion: no new synchronous dataflow/address defect found; v736 retains the
clean one-barrier scheduling change. v737/v738 retain their prefetch but incur
additional compiler barriers and wider temporary copies, whose performance
must be measured. All candidates retain the separate baseline raw-load risk.
