# v739/v740: late-barrier generated-source audit

2026-09-05. The explicit barrier move survives code generation, but **neither
candidate removes its parent's additional compiler-generated synchronization**.
In v739, two consecutive `__syncthreads()` calls now appear after MMA3.

Scope: archived E32 / H7168 / I2048 Stage2, route weights **float32**. This is
a local, read-only comparison of generated C++, not GPU execution, ISA review,
hardware counter collection, performance evidence or an OJ result.

Sources:

- [v739/v740 complete generated log](codex_e32_739_740_codegen.log).
- [v720/v736/v737/v738 parent generated log](../v736_v738/codex_e32_720_736_737_738_codegen.log).
- [Parent generated-source audit](../v736_v738/CODEGEN_AUDIT.md).
- [Independent CPU source/AST and K-lifetime audit](audit_v739_v740_cpu.py),
  executed successfully; Ruff also passed.

All line references below refer to the complete v739/v740 log unless explicitly
identified as parent-log lines. The capture contains compiler log/metadata plus
two complete `SOURCE_BEGIN` / `SOURCE_END` bodies.

## Whole-source comparison

Each extracted generated body was checked against its metadata character count
and SHA256. The four parent bodies and both new bodies all matched.

After normalizing only the generated symbol `stage2_codegen_review_N_kernel`:

- **v737 to v739:** the complete source differs only by moving the early steady
  barrier from after A3 to after MMA3. The already present late barrier remains,
  so the moved barrier is adjacent to it. No other character changes.
- **v738 to v740:** the complete source differs only by moving the early steady
  barrier from after A3 to after MMA3. The later barrier after next Up's shared
  write remains in its original position. No other character changes.

The unified diffs each contain one removed and one added barrier line. Removing
the steady-indented barrier lines from each pair also gives exact whole-source
equality. Thus terminal/epilogue control flow, addresses, vector widths, local
arrays, MMA operand expressions and accumulator order match their respective
parent generated sources, not only their Python ASTs.

| Version | Generated characters | SHA256 |
| --- | ---: | --- |
| v737 | 19482 | `0ce0e2514feaa6de541a0497124f36109b4894b32e78f0cfc87c7a54b915951d` |
| v739 | 19482 | `98ab83dfd5eb2650ba1bbe7c6967fbdecac71b54a833eb7f137fe8f114c41d1f` |
| v738 | 19678 | `c6b9bd5a9e44c0df368dcb832d4f7c3208c9fb05176016bf2d42115fc45ffde1` |
| v740 | 19678 | `611851b08aadd2bb1e9ee7c76ef745358b2f807fbb11b22f0b4fd918ece9f376` |

## Synchronization did not decrease

| Version | Static sites | Steady sites per iteration | Other sites | Positive-row K32 C++ call visits |
| --- | ---: | ---: | ---: | ---: |
| v720 reference | 3 | 2 | 1 terminal | 63 |
| v737 parent | 4 | 3 | 1 terminal | 94 |
| v739 | 4 | 3 | 1 terminal | 94 |
| v738 parent | 5 | 3 | 1 prologue + 1 terminal | 95 |
| v740 | 5 | 3 | 1 prologue + 1 terminal | 95 |

The final column is derived from the captured C++'s 31 steady iterations and
one terminal K64 tile, for a positive-row CTA. It is **not** a measured hardware
barrier count. In particular, this audit does not establish whether the downstream
device compiler merges v739's consecutive calls in machine code.

### v739: adjacent late barriers

The exact captured order is:

```text
60       loop-head barrier
123-127  A3 shared -> up_matrix
129-138  next Up global -> next_up, uint4 loads
139-147  current MMA3, private up_matrix/down_matrix1 only
148      moved explicit barrier
149      previously generated additional barrier, still present
151-153  next_up -> up_shared, uint2 stores
155-164  original next Down global -> down_shared
167      terminal producer/consumer barrier
```

The retained A3/B3 operands are distinct from `next_up` and survive its load.
The two late barriers precede shared overwrite; the next loop-head barrier
protects the new shared data before either operand is read. The early barrier
has disappeared, but the extra barrier has **not** disappeared or fused in this
generated C++ capture. The terminal and epilogue match v737 exactly.

### v740: extra Up-write barrier remains

```text
348      prologue barrier
363-367  current A0 load
368      loop barrier before current B0/B1 loads
426-430  current A3 load
432-441  next Down global -> next_down, uint4 loads
442-450  current MMA3
451      moved explicit barrier
453-462  original next Up global -> up_shared
463      additional compiler-generated barrier, still present
465-467  next_down -> down_shared, uint2 stores
469-473  terminal A0 load
475      barrier before terminal B0/B1 loads
```

This retains the parent's Up-then-Down shared-write order. The next iteration's
A0 may precede its loop barrier because the preceding Up write was already
synchronized at 463; the first iteration is protected by 348. The corresponding
terminal A0 is likewise protected by the final 463. B0/B1 remain after 368 or
475, protecting the later Down shared writes. The generated terminal A0 motion
is inherited from v738, not introduced by moving the explicit barrier.

## Data movement and resource limits

Both candidates retain separate shared regions at offsets 0 and 16384 (32 KiB
total). The declarations remain `out_local[64]` floats, `up_matrix[16]` halves,
two `down_matrix[16]` half arrays, and one `next_up[32]` or `next_down[32]` half
array. Global prefetches remain `uint4` loads while the fragment-to-shared stores
remain `uint2`. No address, fragment size, load width or shared layout changes
relative to v737/v738 were found.

These are source declarations, not physical-register, spill or occupancy
measurements. Both metadata records report `reported_n_regs=null` and
`reported_n_spills=null`, which mean unavailable rather than zero.

## Inherited raw-route load risk is not repaired

The float32 partial-row epilogue loads `routed_expert_weights` before checking
whether that row is valid: **v739 log line 272 before guard 273**, and **v740
line 575 before guard 576**. The indices are not clamped. These epilogues are
exactly their parents' generated epilogues.

Consequently, padded lanes in the final expert can attempt raw reads past the
route array's end. A later zero-output branch does not make an empty raw array
safe. CPU source-epilogue checks or random GPU agreement do not prove these
lowered out-of-range loads harmless. These are not v723 bounds-fix candidates,
and this float32 capture does not establish float16-route generated safety.

Conclusion: both late-barrier transformations are isolated in the captured
generated source; their shared dependencies retain synchronization, but the
hoped-for reduction in additional compiler barriers did not occur here.
