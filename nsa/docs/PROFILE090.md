# Profiling the dominant D64 case

After088/089 regressions, profile retained078 B8/L4096 H1/HQ16 D64
S1 BS16 on C500. This is the largest public078repeat runtime case.
No submission kernel changes. profile_case.py verifies reference before
capture and prints source SHA. mcProfiler metrics selected from installed
show_metrics: AP MMA Duty ratio, VLS Duty ratio, L2C Duty ratio,
ISU stall cycles layout, shared memory access efficiency, Global Memory
Read bytes. Custom capture, per-kernel, counts1, kernel_kernel filter.

profile090 completed. Workload PASS and PROFILE_WORKLOAD_COMPLETE on
both profiler passes, source SHA256
03a7de5719e171fc8f2f1715c1e5129649f1b79076927096a9546bf3390aec9b.
Per-kernel kernel_kernel report: MMA duty relative to AP active5.57%,
shared non-conflict proportion78.95%, L2C duty2.38%, global read
75,350,336 bytes. ISU categories: wsm_stall1,984,588;
vls_pipeline_stall2,077,013; vls_wdata_stall1,028,512; valu_stall0.

Limitations: VLS duty reports0% despite memory traffic/stalls; do not
interpret that as no memory work. Categories are not proven disjoint,
and AP-active denominator is not end-to-end time. RPC shutdown warnings
and timestamp-adjustment warnings occurred. Counters suggest investigating
shared exchange/synchronization and memory issue pressure, not proof of
DRAM saturation or a quantitative speedup bound. Raw per-kernel report,
JSON and workload log archived. No kernel change or OJ score claim.
