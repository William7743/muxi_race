# NSA089 unrolled two-query CTA

Same design as088, but T.unroll(2) instead of T.serial(2). Tests whether
loop control/optimization affects the regression seen for two queries
per CTA. Inputs and results remain independent and fully recomputed.
Only simplified D64/BS16 dispatch changes relative to078.

Syntax PASS; paired089 seed442 d64bs16 directly compares078 (not088).
Completed29/29 PASS. Geometric baseline/candidate ratio0.820224;
sum-time ratio0.771048. Large L16384 case91.1232->118.5664us.
Unrolling does not rescue the two-query grouping; aggregate resembles088
but this is not a same-run direct088 comparison. Reject and keep078.
Raw paired089 logs archived; no OJ score inference.
