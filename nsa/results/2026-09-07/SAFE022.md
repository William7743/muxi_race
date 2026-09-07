# NSA022 S1 safe-memory pass ablation

Generated baseline D64/BS16 source has conditional vector K/V loads with
range checks, despite the block-start branch. Hypothesis: legalizer guards
and associated address arithmetic contribute measurable cost. This is not
a demonstrated bottleneck yet.

Candidate changes only the first two S1 JIT pass configurations from
#115804, adding tl.disable_safe_memory_legalize=True. S>1 is unchanged.
Original causal mask and i_s<=i_t condition remain. No caching or timing-phase
dependent behavior. It assumes nonnegative valid block IDs or the documented
sentinel, and sequence length aligned to block size, as in the selected tests.
Do not promote as a generic unaligned/negative-index-safe implementation.

Python syntax passes. Paired qk six-case seed333 completed, all six pass.

|D|BS|baseline us|NSA022 us|
|---|---|---|---|
|32|16|14.861|14.720|
|32|32|22.016|21.862|
|64|16|29.850|29.709|
|64|32|46.899|45.824|
|128|16|55.987|50.099|
|128|32|80.102|68.826|

Raw safe022.jsonl/log, B4 L1024 H1 HQ16 S1, ABBA four rounds ten calls.
D128 merits broader testing; no broad or OJ gain established yet. Full public109
paired comparison seed334 launched. Use summarizer --profile022 (with a space
between option name and value: --profile 022) to avoid NSA007-specific grouping.
Any future promotion requires broader correctness coverage and explicit fallback
for unsupported shapes. Compiler option acceptance is not a performance claim.
