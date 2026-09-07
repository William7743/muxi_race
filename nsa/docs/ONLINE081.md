# NSA081 online128 padded32

Based on080 experiment, with internal BS=max(32,block_size). Block address
still uses raw block_size. Extra K/V columns zero-filled and score-masked;
no additional tokens enter attention. Query fragment, shared probability,
128 threads; other dispatch unchanged. Syntax and representative S8/D64
source export PASS (8917 chars), resolving080 compile failure.

paired081 seed434 sparse suite launched directly against078; correctness
and timing pending. Added tile work may outweigh thread benefit. Keep078.

paired081 terminal9/9 PASS. Changed S8 cases33.0752->59.8784us and
97.1648->193.7152us (versus078). Clearly slower; not promoted. Raw
results archived. Compilable128-thread layout is not a performance win.
