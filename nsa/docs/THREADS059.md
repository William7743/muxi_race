# NSA059 G32 threads256

Parent057. Only shared simple G32 thread count changes128->256; QK/PV
policies and math unchanged. Hypothesis: additional warps improve work
distribution; risk increased scheduling and shared synchronization cost,
or unsupported GEMM layout. Existing S4 gather256 failure is a different
shape/kernel and does not establish this one's behavior.

g32059 seed410 launched after confirming no active jobs. Pending compile,
correctness and timing; retain057. No OJ score inferred.
