# NSA048 reciprocal normalization

Parent NSA045; gather computes one reciprocal per row then multiplies
scores instead of per-element division. Sparse seed388 all9 PASS.
Candidate S2 B2/B4 12.736/35.994us, S4 B2/B4 19.187/57.805us.
Near parent measurements, no clear performance gain; do not promote.
Finite test success does not establish bitwise equivalence. Raw sparse048
log/JSONL retained. No OJ submission or inferred score.
