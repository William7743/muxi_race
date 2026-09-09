# NSA128: own GPU validation of D64 + S8 combination

## Outcome

The combination now passes its own independent local checks. It is available
as a next manual OJ trial, not as a claimed88-point submission. If127 is
already submitted, preserve that feedback rather than replacing its identity.
No automatic OJ submission was made.

Immutable source: probes/probe_nsa128_d64_bounds_s8_packed_prefetch.py.
SHA256:309e71c21ea3190d38b2ba9ef82cc295bb61b66bebf4aa332c6f757ba83db57a.
The initial "static candidate" header is preserved for exact-byte provenance;
this report supersedes that historical status. Parent127 remains unchanged.

## Measured coverage

| Run | References | Qualified timing samples |
|---|---:|---:|
|S8,seed503,public/current,127+128+129|6/6|18/18|
|Full14,seed509,public/current,127+128|56/56|112/112|
|Extended S8 inputs,8 shapes,3 updates,127+128|48/48|Not timed|

These counts include the controls:target candidate128 contributes2 reference
assertions,full candidate128 contributes28,stress candidate128 contributes24.
The supervisor workflow completed in93.54 seconds,exit0,no yield event.
Cgroup OOM counter remained9. The local device is still16GB/25%-compute C500,
not the full64GB OJ device. Host memory was protected by the same descendant-
aware supervisor validated during127; it does not guarantee peer exclusion.

S8 target medians in microseconds:

| Input mode | NSA127 | NSA128 | Guard-relative128/127 |
|---|---:|---:|---:|
|public|62.89920|60.83072|0.966422|
|current|64.01024|61.42976|0.959207|

Combined target ratio0.9628079963 (~3.72% less time). Values are locally
measured,not OJ times.128/129 generate identical S8 CUDA. Across all14 cases,
only case12 changes versus127;all28 exported source hashes match their paired
records. The D64 changes therefore survive the combination without new code
changes on those paths. Do not assign unchanged-source noise to further gains.

## Correctness cases and boundaries

All extra cases use contiguous FP16 tensors,int32 indices and causal=1.
S=8,G=16. Shape tuples are(B,L,H,D,BS):

```text
(1,1024,1,64,16) (4,1024,1,64,16) (1,1040,1,64,16)
(1,1025,1,64,16) (1,1008,1,64,16) (1,1024,2,64,16)
(1,1024,1,64,32) (1,1024,1,128,16)
```

Three fresh updates exercise random current-block inputs; high-magnitude
Q/K with the current block in the last slot and other slots sentinel; and
zero Q with alternating-sign1024 V and current/previous blocks. For the
first block the last pattern duplicates block0,which the reference handles
as repeated selected slots. No negative/int64 or noncausal diagnostic rows
are counted. The tail case exercises fallback behavior rather than the
fixed16x64 prefetch. Output is NaN-filled before every candidate check and
compared in full with the FP32 reference using atol=rtol=.01.

## Score limitation and reproduction

Using actual OJ109 checker values,127's fixed-parent D64 projection, and this
S8 ratio gives86.0714 overall;case12 does not cross the next integer boundary.
That projection is not an OJ result. Actual best remains109/#14165886.00.
The88-point target remains incomplete and needs larger multi-case gains.

CPU audit,from repository root:

```text
python nsa/tests/audit_nsa128.py
python nsa/tests/verify_nsa128_results.py
```

The GPU workflow used run_nsa128_validation.py under run_exclusive_validation.py
in the existing staged calibration tree. --dry-run lists the exact four stages
without importing GPU libraries. Real execution requires the live supervisor.
It refuses overwriting any old output. Helpers/calibration environment are
the same as127;only the S8 stress helper is new. Archive includes all raw rows,
supervisor output,full log and generated CUDA in results/2026-09-09/sources128/.
Do not submit generated CUDA,tests or profiling output;OJ receives only the
single TileLang candidate source.
