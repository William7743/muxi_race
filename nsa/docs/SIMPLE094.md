# NSA094 runtime full-block mask fast path

Parent078; simplified D64/BS16 only. QK clear_accum=True, then mask
scores only if block_end exceeds query_pos+1. Full historical blocks
need no per-element mask. Partial block mask is identical logically.
Branch uses current block index/query position, not testcase identity,
cached results or benchmark phase. Other kernels and layouts unchanged.

Unlike068 unconditional post-QK mask, this skips mask work for complete
blocks. Branch/control overhead may outweigh savings. Syntax PASS.
paired094 seed446 d64bs16 direct078 running. Must test current-block
edge inputs before any promotion. Completed29/29 PASS, geometric
ratio0.993403 and sum-time ratio0.997376. No overall benefit; not
promoted and no expanded edge sweep warranted. Keep078; no OJ score
claim. Raw comparison and compile logs archived.
