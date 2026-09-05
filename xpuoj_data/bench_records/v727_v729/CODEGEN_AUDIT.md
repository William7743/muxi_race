# v727–v729 Stage1 generated-source audit

Date: 2026-09-05. Scope: E32 Stage1, H=7168, I=2048, 256 threads,
48 token blocks. This is a read-only audit of generated C++ source, not a GPU
correctness result, timing result, or ISA/resource-usage measurement. Numerical
and performance results belong in the separate README.

## Captured sources

All line numbers below refer to
`codex_e32_727_728_729_codegen_stage1_v2.log`, except where explicitly noted.

| Version | Source lines | Characters | Static CTA barrier sites |
| --- | --- | ---: | ---: |
| v727 | 5–414 | 24022 | 9 |
| v728 | 418–854 | 26350 | 9 |
| v729 | 858–1266 | 24039 | 9 |

All three retain separate shared allocations at byte offsets 0 and 16384:
one 16 KiB weight tile and one 16 KiB input tile, totaling 32 KiB. Static
barrier counts and local-array declarations do not establish physical register
usage, spills, active occupancy, or performance.

## v727: retain four Gate B fragments

- `gate_matrix0..3` are independent `half_t[16]` arrays at lines 22–25;
  Up's `weight_matrix` is separate at line 27. The only Gate-fragment writes
  are steady loads at 60/65/70/75 and terminal loads at 216/221/226/231.
  They remain unchanged while the shared weight tile holds Up weights.
- Steady synchronization protects all required edges: initial G/A stores to
  reads at 57; Gate B reads to Up overwrite at 78/79; Up store at 82 to first
  Up B read at 92 via 89; final shared reads to next-K overwrite at 197.
  Terminal equivalents are 212, 234/235, and 246.
- Reading A0 at 86 before the Up write/read barrier at 89 is safe: the input
  tile was already protected by 57 and is not overwritten by the Up copy.
- Steady Gate/Up MMA pairs are 98/107, 126/135, 154/163, 182/191.
  Each accumulator independently retains K16 order 0/16/32/48. The common A
  fragment is not modified between each pair. Terminal pairs are analogous.

## v728: defer only Gate's final K16 MMA

- `gate_tail_matrix` at 437 is separate from `weight_matrix` at 436. Its only
  writes are 527 and 698; it survives until Gate's delayed MMA at 607 and 780.
  The intervening Up B loads do not overwrite the retained Gate operand.
- Steady synchronization is complete at 467, 530/531, 541, and 622;
  terminal synchronization is complete at 637, 701/702, and 713.
- Steady Gate MMA order is 481/500/519/607, and Up order is
  550/569/588/616. Each C still accumulates K16 steps in ascending order.
  The final A3 is shared by Gate3 and Up3 without an intervening A write.
  Delaying Gate3 until after Up0..2 does not change either C's arithmetic order.

## v729: full-source proof against v724

Baseline source: `../v724/codex_e32_724_codegen_stage1.log`.
Both captured sources have 24039 characters and nine static barriers.

The comparison extracted the complete text between `SOURCE_BEGIN` and
`SOURCE_END` and normalized only the generated kernel identifier matching
`stage1_codegen_review_\d+`. All non-barrier lines then matched exactly;
the complete unified diff contained only the removal and reinsertion of one
`__syncthreads();` around the final steady Up MMA loop.

- Final Up B3 is fully loaded at 1035–1039, followed by the moved CTA barrier
  at 1040, then the register-only Up MMA at 1041–1049.
- The loop ends at 1050 with **no automatic late barrier added back**.
  Next-K/terminal shared writes remain protected because every prior shared
  read precedes 1040; the remaining MMA uses only private A3/B/C arrays.
- Terminal barriers at 1064, 1142/1143, and 1149, all copies, fragment loads,
  MMA order, layouts, guards, and the epilogue otherwise match v724 exactly.

The intended early-barrier placement therefore survives code generation.
This proves the inspected source transformation, not a performance benefit.

## Conclusion and limits

No definite source-level shared-memory race, retained-fragment overwrite,
missing producer/consumer barrier, or per-accumulator K16-order error was found.
Block-validity conditions enclosing barriers depend only on uniform block
metadata. The terminal path correctly has no post-Up barrier because no later
shared overwrite exists. Effective output rows retain the existing SwiGLU and
FP16-store path; Stage1 padding remains intentionally unwritten as in the base.

The generated source supports proceeding to independent GPU numerical tests.
It does not replace those tests or prove backend machine-code behavior.
