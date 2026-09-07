# NSA081 online128 padded32

Based on080 experiment, with internal BS=max(32,block_size). Block address
still uses raw block_size. Extra K/V columns zero-filled and score-masked;
no additional tokens enter attention. Query fragment, shared probability,
128 threads; other dispatch unchanged. Syntax and representative S8/D64
source export PASS (8917 chars), resolving080 compile failure.

paired081 seed434 sparse suite launched directly against078; correctness
and timing pending. Added tile work may outweigh thread benefit. Keep078.
