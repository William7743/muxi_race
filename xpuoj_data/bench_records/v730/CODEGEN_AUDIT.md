# v730 versus v727: independent generated-source audit

Date: 2026-09-05. Scope: E32 Stage1, H=7168, I=2048, 256 threads.
This audit used local archived source only; no SSH, GPU execution, or probe
changes were performed. Numerical/performance results are separate.

## Inputs and exact comparison

- Baseline: `../v727_v729/codex_e32_727_728_729_codegen_stage1_v2.log`,
  v727 source at lines 5–414; captured source SHA256
  `c82907fcb3527eef80b713d17b82cfefb1ab5ebe51df9404e6d6f03481e22169`.
- Candidate: `codex_e32_730_codegen_stage1.log`, source at lines 5–414;
  captured source SHA256
  `1d05d1422cc2229dbc3db27c631a55805fe7ac16f8a564bc6302489f59b07e71`.

The complete text between each `SOURCE_BEGIN` and `SOURCE_END` was extracted.
Only generated identifiers matching `stage1_codegen_review_\d+` were normalized.
The complete unified diff contains exactly one added and one removed
`__syncthreads();` line: the existing steady end-K barrier moves before the
last Gate/Up MMA pair. All non-barrier lines match exactly.

The entire terminal/epilogue suffix beginning at `for (int i_18 = 0; ...` is
byte-identical. Both sources contain 24022 characters, nine static barrier
sites, and the same input/weight shared allocations at offsets 16384/0,
totaling 32 KiB. No other generated scheduling or arithmetic change was found.

## Synchronization proof

All line numbers here refer to the v730 log.

- A3 is completely loaded at 169–173; Up B3 is completely loaded at 174–178.
  The moved CTA barrier is at 179.
- Final Gate3 and Up3 MMA blocks follow at 180–188 and 189–197. Their operands
  are private `input_matrix`, retained `gate_matrix3`, `weight_matrix`, and
  the two accumulators. Neither block reads shared memory or rewrites a
  retained operand. “Register-only” here describes source operands, not a
  claim about physical register allocation or spilling.
- The loop ends at 198, with **no automatically reinserted late barrier**.
  Thus other waves may start the next shared overwrite only after every wave
  has completed the prior shared reads; the remaining two MMAs cannot be
  damaged by that overwrite.
- Initial producer/consumer synchronization at 57, Gate-B preservation before
  shared overwrite at 78/79, and Up-store visibility at 89 remain unchanged.
  Terminal barriers at 212, 234/235, and 246 and all terminal MMA order remain
  unchanged. Guard conditions enclosing barriers are block-uniform.

Conclusion: the intended early-barrier change survives compilation exactly;
no new source-level synchronization or retained-fragment hazard was found.
This does not establish numerical correctness or a speedup.

## Existing CPU audit: coverage and limits

`audit_v730_cpu.py` was independently rerun with Python 3.12 and passed. It
checks the exact executable-source/whole-AST move, unchanged terminal and host
paths, tagged operands and K16 accumulation order for K counts including 1 and
112, unchanged copy/load/MMA/barrier counts, and two fresh calls through each
mocked expert-count/route-dtype dispatch.

Limits that must not be mistaken for GPU evidence:

- The serial tag model checks shared-read-to-overwrite ordering. It does not
  model cross-wave producer-write-to-read visibility or infer compiler-added
  barriers; the actual generated-source checks above address those edges for
  this captured shape.
- It does not execute numerical MMA, swizzle/lane mappings, floating-point
  rounding, NaN-poisoned padding, partial/empty expert blocks, or GPU races.
  Host dtype checks are dispatch tests, not FP16/FP32 numerical tests.
- It compares the current v727 file rather than enforcing a frozen baseline
  SHA256. A common mutation of both Python sources could escape a relative
  diff audit. The captured-source hashes above identify this independent
  code-generation comparison; preserve input hashes with future reruns.

No current CPU assertion failure was found, and no audit/probe code was changed.
