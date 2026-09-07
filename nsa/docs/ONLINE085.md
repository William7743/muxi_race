# NSA085 online early V load

Parent078. Move synchronous T.copy(V,...,Vs) directly after K load,
before QK. No new async API; original pipeline configuration unchanged.
Potential overlap versus increased shared-memory live ranges. Mathematical
operations, input indexing and dispatch unchanged.

Python syntax PASS. paired085 seed438 sparse9 compares directly to078.
Completed9/9 PASS. Changed S8 B2/L512:32.8832->33.3568us;
B4/L1024:97.9712->97.8432us. One slight regression, one effectively
parity; no established benefit. Not promoted, no OJ submission. Keep078.
Raw paired085 timing and compilation logs archived.
