# Existing profiling evidence and the88-point requirement

## What the new S8 profile establishes

The peer343 metadata binds its single-window profile to byte-exact NSA129
and the same generated CUDA archived with its paired tests. Shape12,
seed0/public masks,10 warmups,256000000-byte cache flush outside capture,
one captured call match the earlier109 capture settings.

| Recorded counter | NSA109 | Peer343 / NSA129 |
|---|---:|---:|
|Total instructions|6497344|6497344|
|Compute instructions|5446720|5446720|
|Memory instructions|1050624|1050624|
|Global read instructions|138496|138496|
|Global write instructions|16384|16384|
|Private read instructions|0|0|
|Private write instructions|0|0|

Read only1_period0, not the process-wide report. All seven tool records
have isError=false. Earlier unrelated110/111 profiles had suspicious write
counts and are not used to support this comparison.

The observed S8 timing improvement is consistent with better scheduling of
ordinary loads; unchanged counts do not prove that mechanism. In particular,
it is not evidence of fewer instructions, less transferred data or removal
of private spills. These are counts, not time fractions or stall statistics.
Non-simultaneous capture and shared-slice counter isolation remain limitations.
No new GPU measurement was launched by this task for this analysis.

## How much must improve to reach88

Use actual109/#141658 checker-message baselines and kernel times, not
alternative telemetry baselines or a fitted local correction. The checker
floors each point's100*baseline/(baseline+kernel_time), then averages14 points.
The current integer sum is1204;88 overall needs1232, a gain of28.

No single point can contribute28 additional points. Even making the S8
case12 take zero time only raises its82 to the ideal100 and the total to
87.2857. Thus an S8-only improvement cannot meet the full target.

As a scale illustration, applying one common time multiplier to all actual
OJ109 point times needs approximately0.82875264, or17.12% less time, to
reach88. This is a requirement from the score formula, NOT a prediction that
17.12% is available, not a claim about hardware utilization, and not a local
or OJ benchmark result. Rounded checker times also limit threshold precision.

The generated JSON preserves, for every actual point and its exact shape,
time thresholds for+1/+2/+3 integer points and ideal single-point headroom.
This makes future prioritization auditable and prevents attributing integer
score improvements to insufficient or unchanged-source timing differences.

## Consequence for the next GPU work

Finish the pending D64 combination regression before spending more GPU time
on minor S8 variants. NSA123's D64 case5 change and NSA129's S8 change affect
different kernels; their combination still requires its own source/shape and
numeric checks. NSA127/128 have additional unverified changes and must not
inherit the standalone peer129 certification.

Future structural experiments need gains on several scored shapes. Existing
D128 K-prefetch experiments216–225 already showed that eliminating a private
array did not beat the parent; repeating that fix is not a new lead. A new
candidate must have a distinct mechanism and qualified timing evidence.

The resource handoff remains unconfirmed. A read-only check found the peer
resume/profile exit0, no compute workers and no peer SSH session, but we have
not overridden the decision to wait for a reliable handoff. Do not launch a
background GPU workload as part of running the analysis script.

CPU reproduction:

```text
python nsa/tests/analyze_nsa129_profile_budget.py
```

Outputs results/2026-09-09/nsa129_profile_score_budget.json. Source hashes,
capture parameters, counter schema, all14 checker scores and target bracket
are asserted. The full88 goal remains unachieved; actual OJ best is86.00.
