# NSA033 BS16-only CV1

Parent NSA028, with ordinary safe-off CV1 only for block_size16.
BS32 remains parent CV2; shared paths unchanged. No output caching.

Full public109 seed360: exact109 coverage, all PASS.
Geomean speedup vs #115804:1.05438805x, summed latency ratio1.06828637x.
Changed41 geomean1.15497234x, unchanged68 controls0.99802501x.
D12831 geomean1.17434232x, minimum1.05131262x.
Thus the observed NSA032 BS32 regressions are absent in this run.
This is a single full-suite measurement, not an OJ score.

Direct parent NSA028 comparison across all31 public D128 cases completed
seed361 as paired033: all31 PASS, geomean1.02860972x.
Examples of BS16 baseline/candidate us: B1L1024 16.128/15.206,
B2L1024 30.746/28.480, B4L1024 49.395/47.360,
B8L1024 87.834/86.682. Small latency-bound cases remain noisy.
This supports a broad BS16 benefit rather than just one favorable shape.
Independent full109 repeat seed362 completed all109 PASS, exact coverage.
Geomean1.05261737x, summed latency ratio1.06516633x vs #115804.
Changed41 geomean1.14779815x, unchanged68 controls0.99908622x.
Two full runs and direct parent comparison support NSA033 as the new local
performance baseline. This does not establish an OJ score or exhaust input
correctness validation. Preserve NSA028 as a fallback.

NSA034 combines NSA033 with NSA030 shared safe-off CV2. Full public109
seed363 running as public034, not promoted pending evidence.
