# v743 Stage2-only edge-check trace

## Scope and evidence

This is a **boundary/correctness diagnostic under mcTracer**, not a warmup-followed entry benchmark. It directly calls Stage2 kernels, has no Stage1, and contains no dedicated timing warmup or repeated timed entry loop. Cold first calls, compilation/call setup, data preparation and comparison work occur during the run. Do not treat the intervals between these calls as stable kernel scheduling overhead.

Artifacts:

- [edges-1106109.json](edges-1106109.json), SHA256 `dd6babf36411954eb4432ad900c7574c792df0cb06e2a08aaadef87430b42b3c`.
- [codex_v743_stage2_edges_profile.log](codex_v743_stage2_edges_profile.log), SHA256 `f22d60cafb619801bc13d84bd148e57284f5bf2709850e49303b917921e76dd7`.
- Diagnostic implementation: [remote_v743_stage2_edges.py](../../remote_v743_stage2_edges.py).
- The source/actual-codegen checks are documented separately in [CODEGEN_AUDIT.md](CODEGEN_AUDIT.md). Candidate source SHA at test: `67d25409b20cf6417d79375f57edb3770a79fb1a7a619eb8bc3ca9e3b6e0e7ec`.

The trace identifies device0 as C500. Inputs are synthetic E32/H7168/I2048 with raw2373, padded4608 and 36 M blocks; Stage2 grid is (36,56,1), block is (256,1,1). This is a different fixture from the raw4544/padded6144 entry/codegen fixture and is not official OJ data.

Per-block actual row counts from the diagnostic log:

```text
[0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
 79, 80, 81, 95, 96, 97, 111, 112, 113, 127, 128, 128, 1, 128,
 63, 128, 64, 128, 127, 1]
```

The zero-row expert deliberately owns one allocated M128 block. The test includes the M64 threshold, partial/full M128 blocks, multi-block experts, padding, and the final raw token.

## Event identification and integer nanoseconds

Filter `pid=2, ph="X"` and names exactly `stage2_v743_edge_base_kernel` or `stage2_v743_edge_probe_kernel`, then sort by `ts`. There are exactly eight matching device execution events.

The script and log establish their order: FP32 route first, FP16 route second; within each dtype, seeds74301 and74302, each calling v723 baseline then v743 probe. The trace names alone do not encode dtype or seed.

The JSON `ts`, `dur`, and argument timestamps are integer **nanoseconds**. For example, `ts=1788605838601212416` is 2026-09-05 10:57:18.601212416 UTC, consistent with the compile/run log; `dur=1274880` is 1274.880 microseconds. Keep timestamps as integers before subtraction or conversion; floating-point epoch timestamps near 10^18 lose low-order bits. The duration below is the event's `dur`, not `complete_ts-submit_ts`.

| Route dtype | Seed | Kernel | Exact co_id | Start ts (ns) | dur (ns) | dur (µs) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| FP32 | 74301 | v723/base | 143 | 1788605838601212416 | 1274880 | 1274.880 |
| FP32 | 74301 | v743/probe | 151 | 1788605839121716736 | 1136640 | 1136.640 |
| FP32 | 74302 | v723/base | 629 | 1788605839623221760 | 1289472 | 1289.472 |
| FP32 | 74302 | v743/probe | 635 | 1788605839624514304 | 1122048 | 1122.048 |
| FP16 | 74301 | v723/base | 1079 | 1788605854460311296 | 1285632 | 1285.632 |
| FP16 | 74301 | v743/probe | 1087 | 1788605854931721984 | 1136384 | 1136.384 |
| FP16 | 74302 | v723/base | 1529 | 1788605855324784384 | 1281792 | 1281.792 |
| FP16 | 74302 | v743/probe | 1535 | 1788605855326068992 | 1124096 | 1124.096 |

All four recorded probe durations are lower than their baseline counterparts. This is a description of these eight correctness calls only: their order is not counterbalanced, they have no dedicated timing warmup, and tracer/cold-call effects are present. No stable speedup percentage, entry-performance estimate, or OJ prediction is assigned to this table. Independent untraced entry repeats belong in the experiment README, not in this diagnostic table.

## Exact resource metadata

The following values are copied from `args.mem`, not inferred from Python buffers or C++ array declarations. Every event in each row agrees.

| Route dtype / implementation | co_id values | registers_per_thread | dynamic_shared (bytes) | static_shared (bytes) | private_per_thread (raw) | private_total (raw) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FP32 v723/base | 143, 629 | 154 | 32768 | 0 | 0 | 0 |
| FP32 v743/probe | 151, 635 | 154 | 32768 | 0 | 0 | 0 |
| FP16 v723/base | 1079, 1529 | 154 | 32768 | 0 | 0 | 0 |
| FP16 v743/probe | 1087, 1535 | 154 | 32768 | 0 | 0 | 0 |

Other shared raw fields, identical for all eight events:

```text
block = {"x":256,"y":1,"z":1}
grid = {"x":36,"y":56,"z":1}
max_block_size = 256
device_id = 0
queue_id = 0
hw_queue_id = [872420832]
is_dynamic_parallel = false
is_recompiled = false
mtreg_occupancy(%) = 30
shared_memeory_occupancy(%) = 50
```

For **each dtype**, this trace reports no increase in registers/thread, dynamic/static shared memory or the two private-memory fields for v743 versus v723. It independently confirms that the runtime dual path still reports 32KiB dynamic shared, despite using only the lower64 Up rows in tail CTAs. No event-to-event resource inconsistency was found in these eight records.

This supplies resource metadata that the earlier codegen accessor reported as null; it does not retroactively turn null into zero. An older different-fixture Stage2 trace reporting152 registers/thread cannot be used as this candidate's matched baseline: both implementations report154 here.

Interpretation limits:

- `mtreg_occupancy(%)` is retained as a raw named field, **not** identified as measured SM-active warp/CTA occupancy. The misspelled `shared_memeory_occupancy(%)` is likewise not a shared-bandwidth measurement.
- Private fields0 do not prove absence of every spill/local-memory instruction or every form of spill traffic. They are this tool's reported metadata; no instruction-level or hardware-counter spill analysis was collected.
- Equal register/shared reports do not establish identical residency, cache state or scheduling, and cannot attribute a latency change to occupancy.
- This trace provides no actual HBM/shared bandwidth, cache-hit or stall counters. It cannot establish a bandwidth-stall or occupancy cause.

## Boundary results accompanying the trace

All four checks in the log agree:

| Route dtype | Seed | finite | max_abs_full | nonzero_diff | bitwise_equal | padding_nonzero |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| FP32 | 74301 | True | 0 | 0 | True | 0 |
| FP32 | 74302 | True | 0 | 0 | True | 0 |
| FP16 | 74301 | True | 0 | 0 | True | 0 |
| FP16 | 74302 | True | 0 | 0 | True | 0 |

Each seed supplies a new random Up input and routes. Down weights are diagnostic random data generated with seed74300. Invalid Up rows and both output buffers are deliberately filled with NaN before computation; both outputs are then checked finite, bitwise equal, and correctly zero-padded. The script also asserts that the final valid token is not unexpectedly all-zero. Torch is used by this external diagnostic for input generation/comparison, not by the submitted TileLang computation.

These are Stage2 comparisons against v723, not an independent mathematical reference, full run_kernel entry proof, or OJ acceptance. They support the checked branch/padding boundaries without broadening the performance claim.

## Reproduce the local trace extraction

Run from the repository root; this only reads the saved JSON and does not launch GPU work.

```powershell
@'
import json
from datetime import datetime, timezone
from pathlib import Path

path = Path("xpuoj_data/bench_records/v743/edges-1106109.json")
events = json.loads(path.read_text())["traceEvents"]
names = {"stage2_v743_edge_base_kernel", "stage2_v743_edge_probe_kernel"}
kernels = sorted(
    (e for e in events
     if e.get("pid") == 2 and e.get("ph") == "X" and e.get("name") in names),
    key=lambda e: e["ts"],
)
assert len(kernels) == 8
for index, event in enumerate(kernels):
    assert type(event["ts"]) is int and type(event["dur"]) is int
    args = event["args"]
    dtype = "FP32" if index < 4 else "FP16"
    seed = 74301 + (index % 4) // 2
    kind = "base" if index % 2 == 0 else "probe"
    assert f"_{kind}_" in event["name"]
    print(dtype, seed, kind, args["co_id"],
          "ts_ns", event["ts"], "dur_ns", event["dur"],
          "mem", args["mem"], "block", args["block"], "grid", args["grid"])
seconds, fraction = divmod(kernels[0]["ts"], 10**9)
print(datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
      + f".{fraction:09d} UTC")
'@ | python -
```
