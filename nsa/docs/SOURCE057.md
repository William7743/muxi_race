# Generated-code inspection of057

Recent shared simple micro-experiments058..063 did not provide compelling
gains. Export057 G32 BS32 compiler code for D32/D64 to inspect shared
allocation, synchronization and load lowering before another change.
Export helper now accepts groups/block-size while preserving defaults.
source057g32 launched. Generated code is inspection only, never executed
as an external submission kernel. Pending export, no bottleneck claim.

Export completed. D64 source retained locally in results/2026-09-07.
Observed launch_bounds128; Qs/acc_cast/reduction workspace alias offset0,
Vs offset2048, Ks offset4096. Q/K loads use uint4 (16 bytes), not scalar
loads. K has explicit bounds predicates. Thus manual shared aliasing and
simple load-vectorization are not obviously missing optimizations.
These source observations do not measure occupancy or establish a hardware
bottleneck; generated MMA builtin is compiler output, not injected code.
