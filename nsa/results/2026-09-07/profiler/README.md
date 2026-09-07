# Hardware profiling

Tool discovered: /opt/mcProfiler-ubuntu18.04/mcProfiler, version 3.8.1.4.
Initial target: baseline vs NSA002, B4/L1024/H1/HQ16/D64/S8/BS16.
The workload checks correctness with seed2718, warms up, then optionally enters
an explicit profiler region. No profiler calls are added to submission code.

Initial global-mode collection completed for baseline and gather. Files named
`baseline_report.json` and `baseline_task.json` are aggregate diagnostics, NOT
validated per-kernel measurements. Their scope includes more than the expected
bounded target launches (umd_data is empty), so no kernel bottleneck conclusion
is drawn from those counters.

Per-kernel collection completed for both variants. Both workloads passed the
full reference check before capture (seed 2718) and printed
PROFILE_WORKLOAD_COMPLETE. Raw collection logs and reports are retained.

| Selected report | Total cycles (Kcycles) | Global read bytes | Achieved waves (count) |
|---|---:|---:|---:|
| baseline kernel_kernel |1836.69|9553984|4021|
| NSA002 gather_kernel_kernel |2001.77|9568768|4018|

These are instrumented diagnostic counters, not comparable to uninstrumented
event latency. The collected waves do not exactly match the expected 4096
one-wave blocks; task JSON still contains an empty umd_data section and global
counters. Thus report attribution/completeness is not sufficiently validated
to infer occupancy, shared-memory pressure or the full latency regression.
In particular, Achieved waves is a COUNT, not an occupancy ratio. No improvement
or hardware bottleneck is claimed from these numbers.

Reproduce on the GPU host using `bash nsa/tools/run_profile.sh baseline` or
`bash nsa/tools/run_profile.sh gather`. The wrapper keeps collection.log under
nsa/results/profile-<timestamp>-<candidate>. mcProfiler itself emits reports
under its installation directory; the final log line supplies that path.
Copy reviewed reports into nsa/results for archival. No OJ submission code is
modified by profiling. Bash syntax and Python compilation are checked locally
or remotely as applicable; this is not a guarantee of metric accuracy.

Instrumented timing is not automatically comparable to the uninstrumented ABBA
benchmark. The first CLI attempt without --kernelname did not launch the task;
that task-local profiler process was stopped, with no user processes stopped.
