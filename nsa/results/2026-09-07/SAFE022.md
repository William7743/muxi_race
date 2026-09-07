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

Python syntax passes. Paired qk six-case seed333 test launched; results pending.
Any future promotion requires broader correctness coverage and explicit fallback
for unsupported shapes. Compiler option acceptance is not a performance claim.
