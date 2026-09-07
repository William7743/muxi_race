# NSA036 shared vec16 layout

Parent NSA034. Only annotate Qs/Ks/Vs in shared safe-off chunk with
make_mma_swizzle_layout(vecSize=16). Math and tile dimensions unchanged.
Chunk seed369 both correctness checks PASS. BS32 65.587us vs original
baseline80.243us, but slower than NSA034's approximately59us in earlier
paired runs. BS16 is unchanged control. No reason to promote or spend a
full-suite run on this candidate. This does not prove a bank-conflict cause.
Raw layout036 JSONL/log retained. No OJ submission.
