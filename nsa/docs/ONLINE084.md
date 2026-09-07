# NSA084 online reciprocal epilogue

Parent078. After the online loop, compute reciprocal ls once per row,
then multiply output columns instead of dividing each element. All loop
operations, layouts, masks and dispatch are unchanged. Potential reduction
in epilogue work; compiler may already optimize divisions equivalently.

Python syntax PASS. paired084 seed437 sparse9 direct078 comparison
completed9/9 PASS. Changed cases33.0496->32.896us and
97.3696->97.2416us: less than0.5% difference, insufficient evidence
of improvement. Do not promote; keep078. No OJ score inferred.

Representative S8/D64/BS16 generated code differs only at the epilogue:
one ls reciprocal followed by sixteen multiplies replaces sixteen
divisions. Export8063 characters versus078's8018. Source-level difference
is real, but machine-level optimization and timing benefit are unproven.
Both raw timing and exported source are archived under results/2026-09-07.
