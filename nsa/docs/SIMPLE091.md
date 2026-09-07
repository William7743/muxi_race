# NSA091 transposed shared K

Following profile090 shared-memory efficiency clue, duplicate simplified
kernel for D64/BS16 only: Ks[D,BS], parallel store K[n,d] into Ks[d,n],
then QK GEMM transpose_B=False. This preserves the logical dot product.
Query, V, softmax, thread count and output path unchanged. Other dispatch
paths retain078. Potential lower shared conflicts versus transpose-store
overhead; no assumption that profiler alone proves a gain.

Syntax PASS. paired091 seed443 d64bs16 suite directly compares078.
Completed29/29 PASS. Geometric baseline/candidate ratio0.774136,
sum-time ratio0.690548. L16384 case91.4688->141.6064us. Reject.
Changing K storage direction does not improve this implementation; no
counter comparison was collected, so do not claim conflict rate improved.
Raw paired091 logs archived. Keep078; no OJ submission.
