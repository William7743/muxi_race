# v720 / v731 / v732: mcTracer stage timing and resource metadata

Capture date: 2026-09-05. This is a **constant-input local diagnostic with
tracing enabled**, separate from the untraced random-input result in
[README.md](README.md). It is not OJ feedback or a hardware-counter bandwidth test.

## Evidence and fixture

- Trace: [concat-1095033.json](concat-1095033.json), SHA256
  `d7e52d9dad5f50b1c055622fdbb2ec01311abb23cdf2860eb8d1d7f32dc53e79`.
- Output: [codex_profile_731_732_20260905.log](codex_profile_731_732_20260905.log),
  SHA256 `bf51ae80638d5f7a4a51e5bc47130d9ba5f5a098a94842a5fa8b2e6d53efa431`.
- Candidate order: c0=v720, c1=v731/K64, c2=v732/K32, using the tested attempt-2
  source hashes in README, before result-only header updates.
- Local C500; E32/H7168/I2048, alternating64-220 routing, raw rows=4544,
  padded rows=6144, M blocks=48. The helper allocates constant inputs/weights
  and **FP32** route weights. This distribution is not verified OJ input data.
- One correctness round compares full and entry results against c0, then one
  complete entry warmup and two timed rounds (one entry per round, forward
  then reverse order). Every entry still launches Stage1 and Stage2.
- The constant-input comparisons report max_abs=0 and bad=0/44040192. They are
  not the three random-input correctness repetitions recorded separately.

Filter `pid=2, ph="X"` with names
`stage[12]_stage_ab_case_candidate_<id>_kernel`, sort each candidate by its
integer timestamp, and pair Stage1 with its following Stage2. There are
12 matching events for c0 (six pairs), 10 for c1 and c2 (five pairs each).
Only each candidate's **last two pairs** belong to the timed rounds; reference,
correctness and warmup pairs are excluded.

## Integer nanoseconds, not the generic Chrome-trace microsecond convention

The actual `ts`, `dur` and `args.submit_ts` values are integers in **ns**.
For example, c0's first timed Stage1 has
`ts=1788597818586950400` and `dur=2930944`. Dividing the timestamp by 10^9
places it at 2026-09-05 08:43:38 UTC, matching the capture log; the duration is
2930.944 microseconds, not 2930.944 milliseconds. Keep the approximately 10^18
timestamps as integers while subtracting, then convert the differences.

Definitions:

- gap = Stage2.ts - Stage1.ts - Stage1.dur.
- span = Stage1.dur + gap + Stage2.dur, the recorded kernel-start-to-kernel-end
  interval, not the host/event measurement window.
- S1 share = sum(S1.dur) / sum(S1.dur + S2.dur), excluding gaps.

## Last two pairs

Every duration below is in microseconds, exactly converted from integer ns.

| Candidate | Round | Stage1 | gap | Stage2 | span | Logged entry event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| c0 / v720 | 1 | 2930.944 | 26.624 | 1668.096 | 4625.664 | 4658.176 |
| c0 / v720 | 2 | 2941.184 | 2.816 | 1671.168 | 4615.168 | 4671.232 |
| c1 / v731 | 1 | 4850.944 | 26.880 | 1676.544 | 6554.368 | 6592.512 |
| c1 / v731 | 2 | 4794.112 | 33.280 | 1675.264 | 6502.656 | 6542.848 |
| c2 / v732 | 1 | 13912.832 | 29.952 | 1689.344 | 15632.128 | 15667.456 |
| c2 / v732 | 2 | 13904.640 | 30.208 | 1661.696 | 15596.544 | 15628.800 |

Two-round means and shares:

| Candidate | Stage1 us | Stage2 us | gap us | span us | S1 share, excluding gap | S1 / span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 | 2936.064 | 1669.632 | 14.720 | 4620.416 | 63.749% | 63.545% |
| v731 | 4822.528 | 1675.904 | 30.080 | 6528.512 | 74.211% | 73.869% |
| v732 | 13908.736 | 1675.520 | 30.080 | 15614.336 | 89.249% | 89.077% |

The traced constant-run entry means/medians are 4.664704, 6.567680 and
15.648128 ms. Do not substitute these for untraced random medians 4.643456,
6.898816 and 15.448192 ms.

In these samples the regression is concentrated in Stage1: its mean duration
increases 64.251% / 373.720% versus c0, while Stage2 stays near 1.67 ms.
That localizes the elapsed time, but does not identify the responsible native
instructions, bandwidth, register spills or scheduling stalls.

The gap is only an interval between two captured executions, not proven GPU
idle time or Python overhead. In all six timed pairs Stage2 was already
submitted before Stage1 finished. The exact Stage2 submit-minus-Stage1-end
values in ns are c0 [-2927739, -2953893], c1 [-4847701, -4800924],
c2 [-13909987, -13899873]. Therefore these gaps do not support a claim that
Python waited until Stage1 ended before submitting Stage2. Tracer effects,
unrecorded device activity and scheduling remain possible; no causal
attribution is made. The event-minus-span differences are likewise distinct
measurement windows, not an identified removable CPU cost.

## Original resource metadata

All events of a given candidate/stage have identical metadata below.
`registers_per_thread` is copied from the trace, not inferred from source
arrays. Every block is (256,1,1), all static_shared fields are zero.

| Kernel | grid | registers_per_thread | dynamic_shared bytes | private_per_thread | private_total | mtreg_occupancy(%) | shared_memeory_occupancy(%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 Stage1 | (48,16,1) | 248 | 32768 | 0 | 0 | 48 | 50 |
| v731 Stage1 | (48,32,1) | 140 | 32768 | 0 | 0 | 27 | 50 |
| v732 Stage1 | (48,32,1) | 118 | 16384 | 0 | 0 | 23 | 25 |
| All three Stage2 | (48,56,1) | 152 | 32768 | 0 | 0 | 29 | 50 |

The misspelling `shared_memeory_occupancy(%)` is retained from the JSON.
Neither that field nor `mtreg_occupancy(%)` is interpreted here as measured
active SM/warp/CTA occupancy. Zero private fields are their reported values;
they do not establish that all possible native local-memory/spill traffic is
absent. Register-count reduction is not a spill or occupancy proof.

v731/v732 double Stage1's N-grid count versus v720. v732 halves shared memory
per CTA but doubles K-tile iterations compared with v731. These are simultaneous
structural resource/traffic changes. Both have fewer reported registers than
v720 yet execute much more slowly. The experiment rejects the simple inference
that a smaller accumulator/register footprint necessarily improves this entry.

The generated-source audit observes scalar LDS/output accesses and no explicit
unroll pragmas on the small K16/LDS/MMA/output loops. This is an observation,
not proof that native compilation preserves every loop. The separate v733/v734
loop-kind experiments do not modify these tested source hashes or this trace.
This run has no HBM/shared throughput, cache hit-rate, instruction issue or
stall hardware counters; do not label it a fine-grained bottleneck breakdown.

**Conclusion:** retain v731/v732 as correct-but-slower structural experiments,
not OJ candidates. Resource reductions did not translate into speed here.

## Reproduce the parsing without GPU access

From the repository root in PowerShell (Python keeps JSON integers exact):

```powershell
@'
import json, re
from pathlib import Path

p = Path("xpuoj_data/bench_records/v731_v732/concat-1095033.json")
events = json.loads(p.read_text(encoding="utf-8"))["traceEvents"]
kernels = sorted((e for e in events if e.get("pid") == 2
    and e.get("ph") == "X" and re.fullmatch(
        r"stage[12]_stage_ab_case_candidate_\d+_kernel", e.get("name", ""))),
    key=lambda e: e["ts"])
for cid in range(3):
    group = [e for e in kernels if f"_candidate_{cid}_" in e["name"]]
    assert len(group) == (12 if cid == 0 else 10)
    rows = []
    for a, b in zip(group[-4::2], group[-3::2]):
        assert a["name"].startswith("stage1_") and b["name"].startswith("stage2_")
        assert all(isinstance(e[k], int) for e in (a, b) for k in ("ts", "dur"))
        assert isinstance(b["args"]["submit_ts"], int)
        gap = b["ts"] - a["ts"] - a["dur"]
        rows.append((a["dur"], b["dur"], gap))
        print(cid, "S1/S2/gap/span ns", *rows[-1], sum(rows[-1]),
              "S2 submit minus S1 end ns", b["args"]["submit_ts"]-a["ts"]-a["dur"])
    means = [sum(row[i] for row in rows) / 2000 for i in range(3)]
    print("mean S1/S2/gap/span us", *means, sum(means),
          "S1 kernel share %", 100*means[0]/sum(means[:2]))
    for stage in (1, 2):
        samples = [e["args"] for e in group if e["name"].startswith(f"stage{stage}_")]
        keys = ("mem", "grid", "block", "mtreg_occupancy(%)", "shared_memeory_occupancy(%)")
        metadata = {json.dumps({k: a[k] for k in keys}, sort_keys=True) for a in samples}
        assert len(metadata) == 1
        print("stage", stage, "metadata", metadata.pop())
'@ | python -
```
