# NSA075 direct V fragment

Parent057; duplicate simple JIT for D64 S1 BS16 groups>=16, changing only
Vs allocation from shared to fragment. Other dispatch unchanged. Source-first
export B4/L1024/G16 succeeds: Vs is half_t[16], explicit barriers3 instead
of5 in057, generated source6791 versus7816 chars. This proves a source
change, not a speedup. Register pressure/global access may offset savings.

Syntax and representative compilation PASS. d64075 seed426 d64bs16 suite
launched after source export terminal; correctness and timings pending.
No OJ score claim. NSA057 remains preserved.

Completed exact29/29 PASS; reference/candidate geomean0.808795302x,
cumulative0.740561899x. Rejected. Source V load now uses16 scalar half
accesses per thread rather than baseline uint4 loads; this is a plausible
cost contributor, not measured machine-code attribution. Fewer barriers
did not produce a speedup. Raw results and compile log archived.
