# NSA032 ordinary safe-off CV1

Only change from NSA028: ordinary nsa_chunk128_safe_off CV2 -> CV1.
Shared probability path unchanged. Syntax check passed.
Seed357 chunk suite both PASS: BS16 #11580456.563 ->47.859us.
Seed358 direct NSA028 comparison both PASS:
BS16 50.522 ->48.230us; unchanged BS32 control61.440 ->61.478us.
Thus preliminary changed-path latency reduction is about4.5%.
This is not an OJ score or full-suite result.
Full public109 seed359 all PASS, exact coverage. Geomean1.0506872x,
summed latency ratio1.0643607x. However ordinary BS32 regresses:
B2/L512 24.384->28.787us; B1/L1024 23.770->27.507us.
Do not promote this unrestricted variant. Extended edge run completed;
raw edges032.log retained.

NSA033 restricts CV1 to BS16 and restores parent CV2 for BS32.
Full public109 seed360 launched as public033, pending. This is shape-only
dispatch and continues computing from current input on every invocation.
