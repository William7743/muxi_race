# NSA030 shared PV split

Only change from NSA028: nsa_shared_chunk128_safe_off CV4 -> CV2.
Ordinary chunk and all other paths unchanged. Python syntax check passed.

Seed352 chunk suite: both cases PASS. BS32 baseline #115804 80.474 us,
candidate 59.392 us. BS16 is an unchanged-path control.

Seed353 direct NSA028 comparison: both PASS; BS32 60.800 -> 59.392 us,
ratio 1.023707x. This is preliminary paired evidence, not an OJ score.
Full public109 seed354 launched as public030; results pending.
Keep NSA028 as established candidate until full validation completes.
