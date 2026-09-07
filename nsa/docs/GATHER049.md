# NSA049 small-grid gather

Parent NSA045 removes grid>=1024 gate for otherwise unchanged S2/S4
D64 BS16 G16 gather dispatch. Sparse seed389 all9 PASS.
New small-grid S2 11.981->11.149us; S4 15.974->12.070us.
S8 unchanged control24.640->24.589us. Small S2 latency is noisy;
S4 shows stronger initial benefit. Full109 seed390 all109 PASS, exact
coverage; geomean1.06500097x, summed latency ratio1.08516943x.
Profile040 changed/control grouping does not cover new small-grid paths;
do not interpret its control statistics for this candidate.
Extended small-grid edges completed, raw edges049 retained. Independent
full repeat seed391 running as public049repeat. No OJ score inferred.
