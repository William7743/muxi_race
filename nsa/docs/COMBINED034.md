# NSA034 combined PV splits

NSA033 ordinary BS16 CV1 plus NSA030 shared safe-off CV2.
All other paths remain NSA033. Full109 seed363 exact coverage, all PASS.
Geomean vs #1158041.05624632x, summed latency ratio1.07197498x.
Changed41 geomean1.15874547x, unchanged68 controls0.99887974x.
D12831 geomean1.17958003x. No OJ score inferred.

The full-suite difference from NSA033 is small and not sufficient alone
to establish incremental benefit. Direct31 D128 comparison vs NSA033
seed364 completed as paired034: all31 PASS. Five changed shared-path
cases geomean1.02802231x vs NSA033, all five improved. Example us:
B4L1024 60.762->58.995; B2L1024 34.662->33.843.
Extended40 edge checks all PASS, maximum absolute error0.00390625.
Independent full109 repeat seed365 all109 PASS, exact coverage:
geomean1.05717658x, summed latency ratio1.07175336x vs #115804.
Changed41 geomean1.16075533x, unchanged68 controls0.99924561x.
With direct parent comparison and40 edge checks, use NSA034 as local
performance baseline; preserve NSA033 fallback. No OJ score inferred.

NSA035 only unrolls ordinary safe-off QK loop, leaving math unchanged.
Chunk suite seed366 launched as unroll035; pending, not promoted.
