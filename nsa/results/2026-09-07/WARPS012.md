# Two-warp experiment

NSA012: exact #115804 with128 threads for S1 BS32,64 otherwise.
Comparison starts B4/L1024/H1/HQ16/D32. BS16 control passes, but BS32
candidate fails compilation with a layout conflict between acc_s and acc_cast.
This is not a correctness PASS or a performance result for the changed path.
Raw partial result and complete error: warps012.jsonl/log.

The compiler reports differing replication/fragment mappings for QK scores
and the PV input. NSA013 replaces S1 acc_cast storage with shared memory to
bridge them; BS32 remains128 threads and BS16 remains64. BS16 also changes
probability storage, so it is not an unchanged control in NSA013.
Validation/performance tests are pending. S>1 untouched.

NSA013 completes all six full-output checks, seed322, B4/L1024/H1/HQ16/S1.
Same paired ABBA ten-call protocol:

|D|BS|baseline us|NSA013 us|
|---|---|---:|---:|
|32|16|15.10|15.32|
|32|32|21.73|25.06|
|64|16|29.17|29.58|
|64|32|46.31|37.30|
|128|16|55.60|56.49|
|128|32|79.31|65.51|

Shared probability resolves the tested layout conflict. D64/D128 BS32 look
promising, but D32 regresses and BS16 has no gain. This is one seed/session,
not a globally better implementation or an OJ score. Wider coverage/repetition
is needed. Raw results: shared013.jsonl/log.
