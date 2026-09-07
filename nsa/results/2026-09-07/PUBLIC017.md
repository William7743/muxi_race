# NSA017 full public suite, seed328

All109 public cases pass full reference for baseline #115804 and NSA017.
Exact multiset coverage checked: zero missing, zero extra occurrences.
Raw public017.jsonl and public017.log retained. Four ABBA rounds, ten calls
per timed batch. No OJ submission or OJ score prediction.

| Scope | Count | Geomean speedup | Summed latency ratio |
|---|---|---|---|
|All|109|1.008159|1.016765|
|Changed paths|5|1.199244|1.199592|
|Unchanged controls|104|0.999781|1.000329|

Changed-path ratios range1.18022–1.22164; controls0.95866–1.05739.
The summarizer labels these groups changed_007/unchanged_007 because NSA017
has the same changed-shape predicate in this suite, not because these are
NSA007 results. Source is probe_nsa017_combined_shared.py.

The gain is real-looking in the targeted cases but too narrow to justify an
88-point claim. Public coverage is not hidden-OJ coverage (public cases even
include L16384, outside the pasted task's L8192 range). All-masked rows are
not covered. Independent repeat and OJ calibration remain outstanding.

Next experiment NSA018 routes S1 D64 through the existing CK2/CV4 chunked
function, leaving other paths unchanged from #115804. Hypothesis: smaller
shared tiles/output fragments help the much more common BS16 configurations.
Syntax check passed. GPU comparison seed329 completed all four checks (two
D32 unchanged controls, two changed D64 cases). D64 BS16 regresses from29.76
to220.20us; BS32 from46.43 to109.16us. Raw chunk018.jsonl/log retained.
Reject NSA018. NSA019 tests a32-wide output fragment (CV2 forD64 only)
instead of16-wide, to isolate whether fine PV splitting is responsible.
Seed330 paired comparison launched; correctness/performance pending.
