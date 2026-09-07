# NSA066 post-PV normalization

Parent057 shared simple: cast unnormalized exp(score-max) to FP16,
perform PV, divide FP32 output accumulator by row sum. Different rounding
than parent, requiring correctness checks. Tests interaction with G32
PV FullCol unlike earlier020 configuration. More output divisions may
offset scheduling benefit. Initial g32066 seed417 launched; not promoted.
G16 is also changed in experimental source and needs validation or gating.
