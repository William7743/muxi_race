# NSA086 online K/V vec16 swizzle

Parent078; annotate only online Ks/Vs with bundled
make_mma_swizzle_layout(vecSize=16). Query remains fragment. No changes
to math, masking, dispatch, threads or pipeline. Hypothesis: alternative
shared layout can reduce MMA shared read overhead. Earlier simple-kernel
layout results do not establish online-kernel performance.

Python syntax PASS. paired086 seed439 sparse9 directly compares078.
Completed9/9 PASS. Changed S8 cases32.8832->34.176us and
96.9344->102.2976us, about3.9%/5.5% slower. Rejected. Raw
paired086 logs archived; no OJ score inference. Keep078.
