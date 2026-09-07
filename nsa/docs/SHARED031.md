# NSA031 full-width shared PV

Only change from NSA030: shared safe-off D128 CV2 -> CV1.
Hypothesis: reduce loop/copy overhead; risk larger accumulator footprint.
Chunk suite seed356 both correctness checks PASS. Changed BS32 path takes
67.226 us vs original baseline80.013 us, slower than NSA030's approximately
59.39 us in prior measurements. Not promoted; no claim of counter-proven cause.
BS16 uses unchanged ordinary chunk and is a control.
No OJ submission. Raw shared031.jsonl/log retained.
