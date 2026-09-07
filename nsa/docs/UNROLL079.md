# NSA079 online selected-block unroll

Parent078, online T.Pipelined loop replaced with T.unroll(selected_blocks).
Same per-block math/order/validity checks; other kernels and dispatch unchanged.
Hypothesis: constant block-index offsets and reduced loop control may help;
code size/register lifetime may regress. Syntax PASS; source079 S8/D64
representative export running. No timing or correctness claim yet.
NSA078 remains working candidate.
