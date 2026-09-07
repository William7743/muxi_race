# NSA079 online selected-block unroll

Parent078, online T.Pipelined loop replaced with T.unroll(selected_blocks).
Same per-block math/order/validity checks; other kernels and dispatch unchanged.
Hypothesis: constant block-index offsets and reduced loop control may help;
code size/register lifetime may regress. Syntax PASS; source079 S8/D64
representative export running. No timing or correctness claim yet.
NSA078 remains working candidate.

Representative export completed8035 chars. Direct-parent paired079 seed433
completed9/9 PASS. Changed S8 B2/L51233.152->36.5696us and B4/L1024
96.8576->106.5344us (~10% slower). Not promoted; preserve078.
Raw correctness/performance results and compilation log archived.
