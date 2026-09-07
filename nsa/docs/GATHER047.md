# NSA047 bounds-only K/V loads

Parent NSA045. Remove pos<=t only from K/V load conditions, keep both
address bounds and score causal mask. For finite inputs masked future
values receive zero probability; no index-bounds relaxation.
Sparse seed387 all9 PASS, candidate S2 B2/B4 12.864/35.674us,
S4 B2/B4 19.123/57.882us. Near parent measurements, no clear gain.
Do not promote or claim general nonfinite equivalence from finite tests.
Raw sparse047 retained. No OJ submission.
