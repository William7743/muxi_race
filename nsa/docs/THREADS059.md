# NSA059 G32 threads256

Parent057. Only shared simple G32 thread count changes128->256; QK/PV
policies and math unchanged. Hypothesis: additional warps improve work
distribution; risk increased scheduling and shared synchronization cost,
or unsupported GEMM layout. Existing S4 gather256 failure is a different
shape/kernel and does not establish this one's behavior.

g32059 seed410 launched after confirming no active jobs. Pending compile,
correctness and timing; retain057. No OJ score inferred.

g32059 all7 PASS. Candidate us11.392,13.978,19.213,18.867,31.526,
54.310,99.942 in suite order. Near recent128-thread results, no clear
incremental benefit; not promoted. NSA060 tests64 threads instead of256
for G32 with otherwise identical057 settings. g32060 seed411 launched.
