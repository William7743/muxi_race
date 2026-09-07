# NSA063 shared simple row reciprocal

Parent057. After reduce_sum compute one reciprocal per row then multiply
acc_s, instead of per-element division. Compiler may already do this;
rounding may differ. Prior048 was sparse gather, not this shared simple
G32 PV layout. Initial g32063 seed414 launched. Changed G16 path also
requires testing/gating if promising. Retain057 until evidence supports it.
No OJ score inferred.

g32063 completed7/7 PASS. Candidate us12.122,13.786,18.803,18.483,
31.398,53.517,100.339 in suite order. Near recent057 timings; no
compelling incremental benefit. Retain057, do not promote063.
