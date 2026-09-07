# NSA066 post-PV normalization

Parent057 shared simple: cast unnormalized exp(score-max) to FP16,
perform PV, divide FP32 output accumulator by row sum. Different rounding
than parent, requiring correctness checks. Tests interaction with G32
PV FullCol unlike earlier020 configuration. More output divisions may
offset scheduling benefit. Initial g32066 seed417 launched; not promoted.
G16 is also changed in experimental source and needs validation or gating.

g32066 failed layout inference: sm row fragment conflicts with acc_o
FullCol fragment in normalization loop. Not a changed-path correctness pass.
NSA067 copies sm into shared denominator before PV to bridge layouts;
g32067 seed418 launched. Additional synchronization may negate benefit.

g32067 all7 PASS. Candidate us11.251,13.939,19.290,19.302,32.230,
55.142,103.552. Shared denominator resolves compile conflict but shows no
performance improvement over recent057. Do not promote067; retain057.
Seven finite-input checks do not establish all numerical edge behavior.
