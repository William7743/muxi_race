# v752: stopped static composition, not for OJ

Candidate: [probe_v752_v748_e16_runtime_m64_both.py](../../probe_v752_v748_e16_runtime_m64_both.py).
Frozen SHA256: `f3ea9a1ff10aa80d1c983f22ee5dd4639d845efac40ea956ba87442d6e43720a`.

This draft was prepared to add the v751 E16 paths to v748 while preserving
v748's E32/E64 routes. Python syntax and Ruff passed. Independent source/host
composition audit, GPU compilation/correctness/timing and OJ validation were
**not performed**: work stopped when the donor v751's second fixture regressed
about3.42%, with0/4 paired wins. The draft is retained to document the occupied
version number, not promoted or treated as a tested combination. See the
[donor batch evidence](../v751/README.md). No GPU work was launched for v752.
