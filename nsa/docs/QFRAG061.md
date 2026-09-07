# NSA061 Q fragment with shared probability PV

Parent057 shared simple kernel changes Qs from shared to fragment, retains
128 threads and groups32 PV FullCol. Tests interaction unlike earlier
64-thread Q-fragment path. Could save shared traffic or increase register
pressure/layout cost. Currently affects G16 too; initial g32 suite only,
so no promotion without gating or additional G16 validation.

g32061 seed412 launched after no active job confirmation. No performance
or compile-success claim yet. Retain057.

g32061 completed7/7 PASS. Candidate us11.341,14.016,18.906,19.034,
32.026,55.706,102.886. No improvement over recent057 changed-path
18.675,18.496,31.693,53.824,100.262us. Not a direct paired regression
measurement, but no signal justifying promotion. Retain057; do not extend
this Q-fragment change to G16 without a fresh hypothesis.
