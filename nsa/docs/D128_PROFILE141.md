# Current D128 profiling and combined-output experiment

## Source-bound baseline capture

Profile NSA128, case6 (B8,L1024,H1,HQ16,D128,S1,BS32), seed0/public.
The workload checks the full FP32 reference before capture, uses10 warmups,
flushes256000000 bytes outside the window and captures one call per profiler
pass. Its generated CUDA SHA37e9597746b48fbc651e96ff0a322a79320827fb2c3db3ce9fb0176f8dfe6ac6
matches the previously benchmarked D128 source. Parent source SHA is
309e71c21ea3190d38b2ba9ef82cc295bb61b66bebf4aa332c6f757ba83db57a.

Observed hardware remains16GB/25%compute sGPU, not the requested full64GB device.
The guarded profiler workflow exited0 in103.80s. No peer work was stopped;
the host OOM count stayed9. The profiler was terminal before timing141.

| Single-window metric | Reported value |
|---|---:|
|Global read bytes|37,591,808|
|Global write bytes|33,554,880|
|Expected output tensor bytes|33,554,432|
|Private read/write instructions|0 /0|
|Shared non-conflict access proportion|73.4043%|
|MMA duty relative to AP active|11.4025%|
|WSM stall count|3,797,769|
|VLS pipeline stall count|1,599,085|
|VLS write-data stall count|524,288|
|VALU stall count|0|

All selected records report isError=false. Only1_period0 is used, not process
aggregates. The448-byte write discrepancy is retained; its cause is unresolved.
RPC-port and topology warnings appear in the raw log. Source/schema PASS does
not prove exact counter accuracy or isolation across a physical shared device.
Stall categories are not proven disjoint, and AP-active duty is not a fraction
of end-to-end time.73.4% non-conflict accesses does not imply26.6% recoverable
performance. The profile supports investigation, not a speedup prediction.

## NSA141 hypothesis

Use NSA138 as parent (its case6 kernel is unchanged from128). Retain the two
DV64 PV results as FP16 per-lane values until both halves are computed, then
write a full G16xD128 shared tile and one full-row output copy.
Keep the same normalization, GEMMs, casts and exact lane/element mapping.
The first half is stored in8 per-lane FP16 values; the second remains in the
existing local_h. The2048 output-coordinate mapping is bijective.

Unlike rejected direct-output114, this retains a wide shared-to-global copy.
Unlike separately live output staging, the final full tile's lifetime begins
after both V reads. The compiler may alias shared storage, but that is checked
from generated code rather than assumed. Explicit CTA fences protect reuse.
These include an extra full barrier, so this is NOT automatically a reduction
in synchronization cost. Additional live registers or private temporaries may
offset any writeback benefit.

No attention math, pass configuration, external source, async or other path
is changed. Only nsa_d128_scheduler differs; AST and coordinate checks are not
substitutes for runtime correctness and timing.

## Outcome

NSA141 target screen seed593/public+current:4/4 reference assertions pass,
12/12 timing samples qualify under the original guards.141/138 normalized
time ratios are1.36305/1.35832: about36% slower. Reject141; no OJ trial or
full-shape regression is warranted for this failed performance hypothesis.
The screen exited0 in52.23s, no new OOM, and all own GPU jobs are terminal.

Generated host source still launches with6144 dynamic shared bytes. Full output
storage aliases the old storage; capacity growth is not an explanation.
Both captured libraries were inspected after measurement using their exact
source/library hashes. Neither embedded IR has addrspace(5) matches.
This does not prove absence of machine-level spills or runtime private accesses
for141 (which was not counter-profiled). The regression's specific cause remains
unresolved; do not claim memory spilling, register count or barrier cost as proven.
Raw source, host code, timing and IR metadata are preserved.

The baseline profile remains useful evidence to avoid repeating spill-removal
experiments without a measured reason. It is not evidence that all shared-memory
restructures will improve performance.138 stays the optional validated candidate,
133 remains actual OJ best86.64, and88 is not achieved.

## Reproduction commands

```bash
python nsa/tests/verify_profile128_d128.py
python nsa/tests/audit_nsa141.py
python nsa/tests/verify_nsa141.py
```

GPU drivers profile_nsa128_d128.py and run_nsa141_screen.py require the existing
exclusive supervisor and pinned staged helpers. Results remain in
results/2026-09-09/profile128_d128*, and the separate141 screen files.
No automatic OJ submission. Actual highest remainsNSA133/#141741/86.64;
profiling itself does not establish progress in score or achievement of88.
