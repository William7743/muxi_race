# NSA088 two queries per CTA

Prioritization: public078repeat's largest runtime is S1/D64/BS16
B8/L4096 (~168.91us); several other large S1/D64 cases remain near
baseline runtime. This motivates returning from sparse-only tweaks.

Parent078. Duplicate simplified kernel with ceildiv(seq_len,2) CTAs per
batch/head and two serial current-input queries per CTA. Per-query Q,
K,V loads, mask, reductions, GEMMs and output are unchanged. Shared and
fragment allocations reused within the CTA, not results. Tail guarded.
Dispatch only original simplified D64/BS16 path; other paths unchanged.

Hypothesis: reduced scheduling overhead may help large grids, but serial
work and fewer CTAs may hurt. Historical submission comments suggest
multi-token kernels lost; this tests the exact current-parent variation.

Syntax PASS. paired088 seed441 d64bs16 suite directly compares078.
Completed29/29 PASS. Geometric baseline/candidate ratio0.821035;
sum-time ratio0.773805, clear regression. Largest B8/L4096 case
169.024->218.1389us. L8192/16384 cases also regress. Reject serial
two-token grouping for this path; it does not solve the large-grid
bottleneck. Keep078; local aggregate is not an OJ score. Raw logs archived.
