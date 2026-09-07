# Targeted profiling after076

D64 micro-experiments068..076 did not improve057. Collect counters for
the retained D64/S1/BS16 path rather than infer bottlenecks from barriers.
profile_case.py accepts --source and prints its SHA256, preserving old
defaults. Workload correctness precedes capture, no submitted code changes.

profile057d64 started on C500 using mcProfiler perf_exec, custom Total
Cycles, Achieved waves, Global Memory Read bytes; B4 L1024 H1 HQ16 D64
S1 BS16. Three capture launches, kernel filter kernel_kernel, counts1.
Pending output. Need verify kernel attribution and metric units before
interpreting counters; earlier profiler counts were not occupancy percentages.

Completed; workload PASS and PROFILE_WORKLOAD_COMPLETE. Source hash
6ed6c698f7fcd544ec4ff4681f66f4900e0aef9382a9920384af90ff57401419.
Per-kernel report: Total Cycles1743.32 Kcycles; Global Memory Read bytes
9397504; Achieved waves4020 (COUNT, not percent). Expected launch grid4096
blocks, so counter coverage/definition must be clarified; cannot infer
occupancy or bandwidth saturation from these three metrics. Shutdown RPC
warnings occurred but report files were emitted. Raw report/log/data retained.
This collection provides provenance but not a trustworthy bottleneck diagnosis.
