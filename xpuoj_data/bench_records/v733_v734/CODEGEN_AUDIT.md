# v733 / v734: local-loop unroll isolation audit

2026-09-05. Scope: E32/H7168/I2048 Stage1 generated source in
`codex_e32_733_734_codegen.log`, compared with attempt 2 of v731/v732 in
`../v731_v732/codex_e32_731_732_codegen_attempt2.log`.
The [parent audit](../v731_v732/CODEGEN_AUDIT.md) supplies the independently
checked shared-address, slot-mapping, K16-order and output-coverage evidence.
No GPU/SSH/Git operation or probe edit was performed for this audit.

## Exact Python isolation

Whole-module AST comparison passes after restoring exactly five calls from
`T.unroll` to `T.serial`: steady and terminal `ki`, then epilogue `row_tile`,
`col_tile`, `local_id`. No other AST node changes. The complete source from
`import torch` onward is otherwise identical apart from final whitespace;
the leading comments are updated. Outer K, pass dictionaries, copies, barriers,
layouts, emitter internals, dispatch, Stage2 and launch arguments are unchanged.

The existing `audit_v733_v734_cpu.py` was read and rerun successfully, including
its parent geometry/source audit and FP16/FP32 host-dispatch mocks.

| Probe | Audited Python SHA-256 | Generated-source SHA-256 | Characters |
| --- | --- | --- | ---: |
| v733 | `2e1cd0386fdff81cf6bd29d5f1bf244c4b91200d3623d4a5b09d30e480d7da4b` | `400d019e2c3f791a6efde5953bb5ecf8ef60266a9c6026610a72422f186e18ce` | 8339 |
| v734 | `f781e26896180764d193e02d9e437046d855ae8bf2cdac7771547b8a7de1423f` | `59506a076f7b81c51659a0ed4887695170c07144ccc3af153686450282dbe2ad` | 8398 |

Both generated-source hashes were reproduced after removing the one extra
newline added by printing the source. Both kernels compile successfully.

## Actual generated differences

The full generated-source comparison proves the following for each parent/child
pair, normalizing only `stage1_codegen_review_N` kernel names:

1. Exactly five `#pragma unroll` lines are added; total pragma count increases
   from 7 to 12.
2. The epilogue row-valid `if` moves **inside** the `col_tile` loop. Previously
   the nesting was `row_tile -> if(valid row) -> col_tile -> local_id`; now it is
   `row_tile -> col_tile -> if(valid row) -> local_id`.
3. Every other source line is identical after indentation normalization.
   Before the epilogue, all non-pragma text is byte-identical.

For item 2, the exact predicate contains only thread ID, row_tile, group_size,
padded_start and blockIdx.x. It does not depend on col_tile/local_id, performs
no writes, and is unchanged as text. The reordered nesting consequently
selects the same output stores, including no stores for an empty block and
exactly the valid rows of a partial block. This is **not** a claim that the
generated source differs only by pragma lines.

| Source location in the new log | v733 | v734 |
| --- | --- | --- |
| Full source between markers | 6-122 | 128-244 |
| Steady K16 unroll pragma / loop | 46 / 47 | 168 / 169 |
| Terminal K16 unroll pragma / loop | 85 / 86 | 207 / 208 |
| Output row / col / local pragmas | 108 / 110 / 113 | 230 / 232 / 235 |
| Unchanged scalar output expression | 115 | 237 |
| Steady write-to-read barrier | 45 | 167 |
| Steady end-K read-to-overwrite barrier | 68 | 190 |
| Terminal write-to-read barrier | 83 | 205 |

No barrier is removed, moved relative to shared accesses, or added. Static
sites remain three; visible positive-block dynamic counts remain 223 for K64
and 447 for K32. Terminal synchronization and the uniform empty-row guard are
unchanged. The parent audit's cross-wave dependency argument therefore applies
without modification.

All global/shared address expressions, current Gate/Up half selection, local
buffer offsets and sizes, MMA expressions and their reduction order are
unchanged. The output retains Gate slots 0-15/32-47 paired with Up slots +16,
the same FP32 SwiGLU expression and one FP16 store conversion. No address or
math re-derivation is needed to transfer the parent's coverage checks: the
underlying generated expressions are exactly the same, not merely similar.

## Compiler effect and limits

The requested unroll hints survive in the generated C++. They do **not** turn
this capture into fully expanded C++ statements: the five loops are still
printed as loops with pragmas. The emitter's nested A/B loading and MMA i/j
loops still have no explicit unroll pragma. Global-to-shared copies remain
`uint2`; LDS reads and output stores remain scalar in this source.

This capture does not establish whether the native compiler ultimately unrolls
those loops, removes dynamic indexing, spills local arrays, or changes physical
register use. Resource fields `reported_n_regs` and `reported_n_spills` remain
null, not zero. Performance and GPU numerical results belong in separate
experiment records and are not inferred from these hints.

Conclusion: the five-call Python change is isolated, and the generated code
retains the parent's addresses, shared synchronization, paired output math and
coverage. The sole extra lowering change is the semantics-preserving movement
of the row predicate inside the column loop. No source-level correctness
defect was found; this audit does not recommend promoting either candidate.
