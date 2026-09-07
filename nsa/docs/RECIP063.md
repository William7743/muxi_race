# NSA063 shared simple row reciprocal

Parent057. After reduce_sum compute one reciprocal per row then multiply
acc_s, instead of per-element division. Compiler may already do this;
rounding may differ. Prior048 was sparse gather, not this shared simple
G32 PV layout. Initial g32063 seed414 launched. Changed G16 path also
requires testing/gating if promising. Retain057 until evidence supports it.
No OJ score inferred.
