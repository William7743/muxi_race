# v745 Stage1 / chain boundary trace

## What this run establishes

This mcTracer capture accompanies **boundary/correctness checks**, not a dedicated performance benchmark. The diagnostic calls Stage1 directly, checks its workspace, then calls Stage2 for FP32 and FP16 routes using that current workspace. It does not time the normal run_kernel entry, has no dedicated timing warmup, and includes cold calls, compilation/setup, CPU input creation, transfers and GPU comparison work.

Evidence:

- [edges-1110043.json](edges-1110043.json), SHA256 `b3ed870020a7d078441d670bb519921984ca08ea8559b1a06e4fcf5f90c030da`.
- [codex_v745_stage1_edges_profile.log](codex_v745_stage1_edges_profile.log), SHA256 `5bcd6d34924530001ef185af825c45b0049fed694738c7ea4e7e97591a3d01bb`.
- Diagnostic: [remote_v745_stage1_edges.py](../../remote_v745_stage1_edges.py), using the documented metadata from [remote_v743_stage2_edges.py](../../remote_v743_stage2_edges.py).
- Boundary outcome and later untraced entry timing are recorded separately in [README.md](README.md).

The log records the exact files loaded remotely:

| Role | File | SHA256 |
| --- | --- | --- |
| Baseline | probe_v743_v723_e32_stage2_runtime_m64.py | `67d25409b20cf6417d79375f57edb3770a79fb1a7a619eb8bc3ca9e3b6e0e7ec` |
| Candidate | probe_v745_v743_e32_stage1_runtime_m64.py | `12f9dcc12ed1327c6f8eba411bfbee8c39132b0d626818140f8fe15cc7609c96` |

The remote v743 is its original tested-header file, not the current local published-header SHA `5eaa07dc2949351cebcf42373267d4e5d85b906caadd8c37a93dd2d69c6bd0b9` used when cloning v745. The [v743 record](../v743/README.md) documents that restoring the old header exactly reproduces the tested hash and leaves the entire AST unchanged. These identities must not be conflated; the diagnostic baseline is the hash printed above.

After all tests, v745's one static-candidate comment was replaced with three result comments. The final published-header SHA is `ec864ca3ba12de060fd17920ed814f8cc8ba4e415bf28c1a20456a8b3c3cc465`; reversing that header change exactly restores the tested `12f9dcc1...` identity, with unchanged complete AST. This trace used the tested file in the table, not the later header revision.

Local device metadata identifies C500/device0. Geometry is E32/H7168/I2048, raw2373/padded4608, 36 M blocks. This is the synthetic boundary distribution with rows0/1/63/64/65/127/128 and additional intermediate counts, including an allocated zero-row block and a final one-token group. It is neither official OJ input nor the raw4544/padded6144 entry fixture.

## Event order and timestamp units

Select device events `pid=2, ph="X"` named `stage1_v745_edge_{base,probe}_kernel` or `stage2_v745_edge_{base,probe}_kernel` and sort by `ts`. Exactly12 events match.

For each seed, script order is:

1. Stage1 baseline then probe, without a route dtype.
2. Stage2 FP32 baseline then probe.
3. Stage2 FP16 baseline then probe.

The two input seeds are74501/74502. Dtype and seed labels below come from that verified script/log order; they are not embedded in the kernel names.

All `ts`, `dur` and timestamp argument fields are integers in **nanoseconds**. For example, `1788608025473079552` is 2026-09-05 11:33:45.473079552 UTC, matching the run log. Event duration2257152ns is2257.152µs. Arithmetic must retain integer timestamps before unit conversion. Use the event's `dur`; `complete_ts-submit_ts` is not substituted as the kernel execution duration.

| Seed | Stage / route | Role | co_id | Start ts (ns) | dur (ns) | dur (µs) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 74501 | S1 / not applicable | v743/base | 187 | 1788608025473079552 | 2257152 | 2257.152 |
| 74501 | S1 / not applicable | v745/probe | 195 | 1788608025970748672 | 2137344 | 2137.344 |
| 74501 | S2 / FP32 | v743/base | 818 | 1788608042553888256 | 1136896 | 1136.896 |
| 74501 | S2 / FP32 | v745/probe | 826 | 1788608043062901504 | 1119232 | 1119.232 |
| 74501 | S2 / FP16 | v743/base | 1416 | 1788608059658299648 | 1144064 | 1144.064 |
| 74501 | S2 / FP16 | v745/probe | 1424 | 1788608060120921856 | 1121024 | 1121.024 |
| 74502 | S1 / not applicable | v743/base | 1991 | 1788608061531551232 | 2277632 | 2277.632 |
| 74502 | S1 / not applicable | v745/probe | 1997 | 1788608061533831680 | 2154752 | 2154.752 |
| 74502 | S2 / FP32 | v743/base | 2591 | 1788608061538944512 | 1142016 | 1142.016 |
| 74502 | S2 / FP32 | v745/probe | 2597 | 1788608061540120832 | 1116928 | 1116.928 |
| 74502 | S2 / FP16 | v743/base | 3164 | 1788608061547357696 | 1119488 | 1119.488 |
| 74502 | S2 / FP16 | v745/probe | 3170 | 1788608061548480000 | 1121024 | 1121.024 |

These are complete diagnostic samples, not selected warmed-up rounds. Do not derive stable speedup, an entry sum, GPU idle time, host overhead or OJ score from them. In particular, inter-call intervals contain diagnostic/compilation/comparison activity and are not an isolated Stage1-to-Stage2 scheduling gap. The unchanged Stage2 code also has differing observed durations under the two names, illustrating why these cold, fixed-order calls should not be treated as isolated performance attribution.

## Actual resource fields

Direct transcription of `args.mem`; all events grouped in each row agree.

| Stage / route / role | co_id values | registers_per_thread | dynamic_shared (bytes) | static_shared (bytes) | private_per_thread (raw) | private_total (raw) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S1 v743/base | 187, 1991 | 248 | 32768 | 0 | 0 | 0 |
| S1 v745/probe | 195, 1997 | 248 | 32768 | 0 | 0 | 0 |
| S2 FP32 v743/base | 818, 2591 | 154 | 32768 | 0 | 0 | 0 |
| S2 FP32 v745/probe | 826, 2597 | 154 | 32768 | 0 | 0 | 0 |
| S2 FP16 v743/base | 1416, 3164 | 154 | 32768 | 0 | 0 | 0 |
| S2 FP16 v745/probe | 1424, 3170 | 154 | 32768 | 0 | 0 | 0 |

All blocks are (256,1,1). Stage1 grid is (36,16,1); Stage2 grid is (36,56,1). Every event reports device_id0, queue_id0, hw_queue_id[2483033568], max_block_size256, is_dynamic_parallel=false and is_recompiled=false.

Other raw named values:

| Field | Stage1, both roles | Stage2, both roles/dtypes |
| --- | ---: | ---: |
| mtreg_occupancy(%) | 48 | 30 |
| shared_memeory_occupancy(%) | 50 | 50 |

The new Stage1 branches do **not** increase the reported registers/thread or shared/private fields in this matched diagnostic: Stage1 is248 and32KiB for both versions. Both Stage2 route dtypes remain154 and32KiB. No event-to-event inconsistency in the listed resource metadata was found.

Interpretation boundaries:

- This is new runtime trace metadata, distinct from the earlier codegen accessor's null register/spill values. It is not inferred from private-array declarations.
- Equal register/shared fields do not prove identical scheduling, cache state, resource residency or active occupancy.
- `mtreg_occupancy(%)` is retained under its original name, not interpreted as measured SM-active warp/CTA occupancy. `shared_memeory_occupancy(%)` preserves the original spelling and is not measured shared bandwidth.
- The two private fields reporting0 do not prove absence of every spill/local-memory instruction or traffic. No instruction-level spill audit or hardware-counter traffic data was collected.
- No HBM/shared bandwidth, cache-hit or stall counters are provided by these records; there is no bandwidth-stall or occupancy-cause claim.

## Correctness associated with this trace

Two fresh input seeds74501/74502 were used with fixed diagnostic random Gate/Up/Down weight seeds74510/74511/74512. Each new input has NaN in invalid stacked-token rows; both Stage1 workspaces start NaN.

- Both Stage1 valid-row comparisons are finite, max_abs_full=0, nonzero_diff=0, bitwise_equal=True and tolerance_bad=0/4859904.
- Both `padding_untouched_nan` checks are True. Stage1's finite comparison is on the gathered **valid2373x2048 workspace elements**, not the full padded workspace: padding intentionally remains NaN.
- Each seed subsequently exercises FP32 and FP16 routes through the two current Stage1 workspaces. All four chain comparisons are finite, bitwise_equal=True, max_abs_full=0, nonzero_diff=0 and tolerance_bad=0/33030144.
- Each chain reports zero_padding=True and passes the diagnostic's final-valid-token nonzero assertion.
- The final log reports all_tested_bitwise=True.

These are comparisons against v743, not an independent mathematical reference or an OJ pass. The separately called Stage1/Stage2 chain establishes these boundary samples, not normal-entry timing or every run_kernel host path. Torch belongs only to the external diagnostic's input generation and comparisons.

## Read-only extraction

Run from the repository root; no GPU work is performed.

```powershell
@'
import json
from datetime import datetime, timezone
from pathlib import Path

events = json.loads(Path(
    "xpuoj_data/bench_records/v745/edges-1110043.json"
).read_text())["traceEvents"]
names = {
    f"stage{stage}_v745_edge_{role}_kernel"
    for stage in (1, 2) for role in ("base", "probe")
}
kernels = sorted(
    (e for e in events
     if e.get("pid") == 2 and e.get("ph") == "X" and e.get("name") in names),
    key=lambda e: e["ts"],
)
assert len(kernels) == 12
for index, event in enumerate(kernels):
    seed = 74501 + index // 6
    slot = index % 6
    stage = 1 if slot < 2 else 2
    dtype = "N/A" if stage == 1 else "FP32" if slot < 4 else "FP16"
    role = "base" if slot % 2 == 0 else "probe"
    assert event["name"] == f"stage{stage}_v745_edge_{role}_kernel"
    assert type(event["ts"]) is int and type(event["dur"]) is int
    args = event["args"]
    print(seed, stage, dtype, role, args["co_id"],
          event["ts"], event["dur"], args["mem"], args["block"], args["grid"])
seconds, fraction = divmod(kernels[0]["ts"], 10**9)
print(datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
      + f".{fraction:09d} UTC")
'@ | python -
```
