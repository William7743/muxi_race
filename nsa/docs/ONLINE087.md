# NSA087 online K/V vec4 swizzle

Parent078, only online K/V shared layout annotated vecSize=4.
Complements086's vec16 test; math and all other configurations unchanged.
Python syntax PASS. paired087 seed440 sparse9 direct078 comparison.
Completed9/9 PASS. Changed S8 cases33.024->40.1408us and
97.728->120.6912us, about22%/23% slower. Rejected. Both tested
alternative online K/V widths (4,16) lose to078's inferred layout.
Keep078; no OJ score inferred. Raw logs archived.
