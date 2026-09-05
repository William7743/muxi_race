# v720 / v727 / v730: measured mcTracer comparison

2026-09-05, local quarter-C500, MACA 3.7.1, TileLang 0.1.10+maca.
This is diagnostic evidence, not a new OJ score or an untraced performance claim.

## Capture

Raw trace: [interleave-1081234.json](codex_profile_727_730_20260905/interleave-1081234.json).
Run output: `codex_profile_727_730_20260905.log`.
Candidate mapping: c0=v720, c1=v727, c2=v730.
The existing `remote_stage_ab.py` ran E32 / H7168 / I2048, local
`alternating64-220` routing (4544 valid, 6144 padded rows), constant diagnostic
inputs, real `run_kernel` entry, warmup=1, iters=1, rounds=2 forward/reverse.
The tracer was enabled throughout. Random correctness and untraced timing are
separate: [experiment records](../v727_v729/README.md).

Select `pid=2, ph=X`, matching stage1/stage2 candidate kernel names. There are
six complete launch pairs for c0 and five each for c1/c2; take the final two
pairs, excluding reference, correctness, and warmup calls. Integer `ts`, `dur`
and `submit_ts` in this trace are **nanoseconds**. Compute differences as Python
integers before dividing, not floating-point epoch timestamps.

## Final two measured pairs

Units: microseconds. `gap = S2.start - S1.end`; `span = S1 + gap + S2`.
Gap is not automatically CPU overhead or whole-device idle time.

| Candidate | Round | Stage1 | Stage2 | gap | span |
| --- | ---: | ---: | ---: | ---: | ---: |
| v720 | 1 | 2926.848 | 1667.328 | 27.904 | 4622.080 |
| v720 | 2 | 2935.040 | 1669.888 | 27.648 | 4632.576 |
| v727 | 1 | 2933.760 | 1681.920 | 26.880 | 4642.560 |
| v727 | 2 | 2905.600 | 1659.904 | 38.656 | 4604.160 |
| v730 | 1 | 2956.032 | 1673.472 | 26.624 | 4656.128 |
| v730 | 2 | 2929.152 | 1680.384 | 27.904 | 4637.440 |

Mean S1/S2/gap/span in us:

| Candidate | Stage1 | Stage2 | gap | span |
| --- | ---: | ---: | ---: | ---: |
| v720 | 2930.944 | 1668.608 | 27.776 | 4627.328 |
| v727 | 2919.680 | 1670.912 | 32.768 | 4623.360 |
| v730 | 2942.592 | 1676.928 | 27.264 | 4646.784 |

v727's S1 mean is 11.264 us lower, but its two samples bracket the corresponding
baseline values. v730's early barrier does not provide a consistent improvement.
Two traced constant-input rounds cannot establish a small stable speedup.
The measured event-window means (4664.448/4658.944/4683.904 us) are not these
kernel spans and must not be mixed into the same statistic.

## Actual resource metadata

All events within a candidate/stage report identical metadata:

| Kernel | registers_per_thread | dynamic_shared bytes | private_per_thread | private_total | mtreg_occupancy(%) raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| v720 Stage1 | 248 | 32768 | 0 | 0 | 48 |
| v727 Stage1 | 252 | 32768 | 0 | 0 | 49 |
| v730 Stage1 | 252 | 32768 | 0 | 0 | 49 |
| All Stage2 | 152 | 32768 | 0 | 0 | 29 |

Static shared is 0 throughout. Block=(256,1,1); Stage1 grid=(48,16,1),
Stage2 grid=(48,56,1). Raw `shared_memeory_occupancy(%)` is 50 for all.
These vendor metadata fields do not establish achieved SM occupancy, no-spill
behavior, shared bandwidth, or a register-driven performance explanation.

**Trace-only decision:** v727 has only a weak signal; v730 has no proven
incremental gain. The later untraced second routing fixture made v727 about 1%
slower; the combined experiment decision is therefore not to prioritize either
for OJ. Do not replace the verified 80.33-point baseline on this evidence.
Hardware bandwidth/cache/stall counters have not been collected by mcTracer.

## MCPTI inventory and empty-group smoke

`codex_mcpti_inventory_group_20260905.json` is a separate successful invocation
of `remote_mcpti_inventory.py --group-smoke`, not part of the trace:
166 metrics enumerated; context query, empty group creation, and destruction
returned 0. `collection_enabled=false`. This proves only those API operations;
it does not prove event enable/read, replay, counter accuracy, or usable units.
The initial inventory in `../v727_v729/` does not include this later smoke result.

The subsequent [event3 preflight](../v727_v729/codex_mcpti_event3_preflight_20260905.json)
also successfully added an event and queried raw attributes. It deliberately
stopped before enable: scope/instance payload contracts and context-only
coverage were not established. No kernel counter values were collected.
