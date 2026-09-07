# NSA011 built-in cython backend

All four full-output checks pass. Fresh seed320, exact #115804 mathematics,
complete run_kernel timed with four ABBA rounds and ten calls per batch.
The cython backend is substantially slower on these local small cases
(roughly42-59us versus12-14us baseline). Reject this backend change.
Raw timings are in cython011.jsonl/log. No OJ use or score inference.

Next NSA012 changes only S1 BS32 kernel thread count from64 to128, keeping
BS16 and S>1 unchanged. Compiler partitioning/correctness/performance must
all pass before promotion; exploratory tests pending.
