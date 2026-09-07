# NSA030 shared PV split

Only change from NSA028: nsa_shared_chunk128_safe_off CV4 -> CV2.
Ordinary chunk and all other paths unchanged. Python syntax check passed.

Seed352 chunk suite: both cases PASS. BS32 baseline #115804 80.474 us,
candidate 59.392 us. BS16 is an unchanged-path control.

Seed353 direct NSA028 comparison: both PASS; BS32 60.800 -> 59.392 us,
ratio 1.023707x. This is preliminary paired evidence, not an OJ score.
Full public109 seed354 completed: all109 PASS, exact public coverage;
geomean speedup1.0490468x vs #115804, summed latency ratio1.0652958x.
Changed41 cases geomean1.1393061x, unchanged68 controls0.9981185x.
Extended edge validation all36 PASS, maximum absolute error0.00390625.
Includes D32 large-grid vec16, S2 gather sentinel/duplicate/two-valid blocks,
G8 fallback, G32, scaled QK and zero-query checks. These tests still require
at least one valid causal token, so do not establish all-masked behavior.
Independent public109 repeat seed355 launched as public030repeat.
Keep NSA028 as established candidate until repeatability is checked.
