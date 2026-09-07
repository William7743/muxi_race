# NSA023 combination validation

## Full public109, seed337

All109 pass for exact #115804 and NSA023; zero missing/extra occurrences.
Raw public023.jsonl/log. Geomean1.030425x; summed latency ratio1.043529x.
Changed31 cases geomean1.118109x (range0.99152–1.31232); unchanged78 controls
geomean0.997517x. Full suite gain is modest, not evidence of an88 OJ score.
Large-grid edge extension now launched. Independent repeat remains pending.

Boundary run completes: all16 checks PASS, maximum absolute error0.001953125
against FP16-cast full reference. Raw edges023.log. These small shapes do not
exercise the large-grid shared-probability dispatch. Added --include-large
to include B2 L1024 H1 HQ16 D128 S1 BS32, not yet run. Full public109 paired
comparison seed337 has now launched, results pending.

UPDATE: Six-case qk comparison seed336 completes, all six PASS. B4 L1024
H1 HQ16 S1. D128 BS16 baseline55.1168us/candidate50.3936us;
BS32 baseline81.6256us/candidate62.0032us. Raw combined023.jsonl/log retained.
Four ABBA rounds, ten calls per batch. Boundary validation now running;
full public suite and repeatability still pending. Original planning notes follow.

Parent NSA017 passed public109. NSA022 showed D128 local gains in six-case
screening; its full public109 run is still in progress. These gains cannot
be added arithmetically to NSA017's gains.

NSA023 adds copies of the ordinary and shared-probability chunk functions
with only tl.disable_safe_memory_legalize=True changed. Selection applies
only to S1 D128 with sequence length divisible by block size and G>=16.
G8, simplified, S>1 and other paths retain NSA017 dispatch. Valid block IDs
or the contract sentinel are assumed. Unaligned shapes keep the parent's
ordinary legalizer path; no new general correctness guarantee is implied.

Python syntax check passes. GPU validation is pending, queued conceptually
after public022 finishes (no concurrent benchmark launched). No OJ submission.

Prepared tests/validate_edges.py for post-benchmark correctness validation:
four shapes including G8/G32 and sentinel-padded S4, each with current block,
first block, Q/K scaled by4, and zero-query modes. Every row has at least
one valid token; all-masked semantics remain outside this check. NaN-filled
output detects missing stores. Full FP32 reference comparison, no timings.
Syntax passed; GPU execution pending. Test preparation is not a PASS result.
