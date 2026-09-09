# NSA128: isolated peer S8 lead, pending GPU validation

## Provenance and scope

Parent: probe_nsa127_combined_d64_bounds.py, SHA256
ca7e74e935cf6e2a53e1880f1d8a3cd3a43055482c95d2b8ef030f6555acbf22.

Candidate: probe_nsa128_d64_bounds_s8_packed_prefetch.py, SHA256
309e71c21ea3190d38b2ba9ef82cc295bb61b66bebf4aa332c6f757ba83db57a.

Source of the new component: peer workspace submissions/nsa_v342.py, SHA256
8aabb78cc1dfbcc847e4b03cc79587d93100c97d7a3a055df149eedc3b58fb6a.
Peer318's original nsa_online_direct_output body matched our parent exactly.
Only that factory body was transplanted; peer D128 compiler flags, scheduling
changes and wrappers were NOT copied. Other factories and dispatch remain
identical to127. All these sources compute from current inputs each call.

## Design

The existing S8/H1/G16/D64/BS16/aligned-sequence specialization uses64 lanes.
Each lane preloads16 FP16 V values into eight uint32 registers, before K load
and QK GEMM. After GEMM it extracts both16-bit halves into the existing shared
buffer. The mapping covers every element of the16x64 V tile exactly once.
This uses ordinary TileLang loads/reinterpretation, not async copies, external
device code, extra compiler flags or cached results. GEMM and synchronization
calls are unchanged. Generated instruction width, packing convention and
shared-buffer lifetime still require device-source inspection and GPU tests.

audit_nsa128.py reconstructs the exact three source edits from127 and checks
AST equality, specialization guards, complete lane coverage and exhaustive
16-bit paired-word recovery. It is a CPU audit, not a GPU correctness proof.

## Peer evidence, not candidate evidence

Raw peer data: results/2026-09-09/peer_v342_confirm.jsonl.
Case12 only, seeds42/137, public/current masks, three modules318/321/342,
seven rounds:12 reference assertions and84 guarded timing samples. Reported
peer342 vs318 guard-normalized geometric ratio is about0.96273.
Medians in microseconds (342 /318):

| Seed/mask | Peer342 | Peer318 |
|---|---:|---:|
|42/public|60.63616|62.92992|
|42/current|61.49632|63.91808|
|137/public|60.70272|62.91968|
|137/current|61.52704|63.92832|

These records justify testing the component, not promoting NSA128. They were
produced by the peer, not by this task. No NSA128 GPU result or OJ score exists.
Our16GB slice is not the full64GB OJ device.

## Interrupted handoff attempt and next steps

Peer342 final script completed with exit0 and no compute Python visible.
We then launched the single-process NSA127 validation driver. It reported a
32GiB cgroup limit and1914925056 bytes used, then stopped during TileLang
initialization. No new stress/full-suite/source-export results were produced.
The log is preserved as nsa127_batchgap_init.log. Subsequent observations
showed peer343 continuing and the cgroup OOM counter increasing from7 to9.
Kernel logs were inaccessible; exact victim attribution remains unavailable.
No peer task was signalled or terminated by us.

An individual script's completion is not a handoff if the worker auto-starts
the next version. User says wait, and currently does not know whether the
other worker is finished. Do not launch into another idle gap. The new driver
requires --handoff-confirmed for actual execution, refuses artifact overwrite,
and checks24GiB host-memory headroom before importing GPU libraries. These
checks are not a shared lock and cannot stop another worker from starting.
The flag must not be supplied without a reliable resource handoff.

Offline checks:

```text
python nsa/tests/audit_nsa128.py
python nsa/tests/test_nsa127_validation_driver.py
python nsa/tests/run_nsa127_validation.py --dry-run
```

The driver targets the existing staged server test tree; it depends on the
calibration/source-dump helpers documented in D64_124_127.md. It is not a
standalone repository-root GPU command. After handoff, finish127 validation
first, then independently compile, inspect, stress and measure128 against127
with fresh inputs and guarded paired timing. Preserve all raw outputs.
Until then keep123 as the prior locally verified optional trial and109 as
the actual OJ best (86.00). No automatic OJ submission was made.
