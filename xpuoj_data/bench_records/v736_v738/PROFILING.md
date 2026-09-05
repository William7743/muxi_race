# v720 / v736 / v737 / v739: measured Stage2 tail comparison

2026-09-05 local mcTracer diagnostic. **v736's approximately 1.3 ms regression
is in Stage2**, not Stage1 or the gap between the kernels. This capture does not
establish the underlying hardware cause or a new OJ score.

## Capture and selection

Raw trace: [tail-1101447.json](tail-1101447.json).
Run output: [codex_profile_736_737_739_20260905.log](codex_profile_736_737_739_20260905.log).
The file names cover the surrounding experiment series; this particular capture
contains **c0=v720, c1=v736, c2=v737, c3=v739**, not v738.

E32/H7168/I2048, 4544 valid / 6144 padded tokens, 48 row blocks; local
`alternating64-220` routing and **constant diagnostic inputs**. This fixture is
not a verified reconstruction of private OJ scoring inputs. The real
`run_kernel` entry is timed, warmup=1, iters=1, rounds=2 forward/reverse, with
the tracer active. Both the precompiled full closure and the entry pass the
baseline comparison, with the log printing `max_abs=0.000000, bad=0/44040192`.
The checker tests finiteness and `abs(diff) <= 0.05 + 0.05*abs(reference)`;
the maximum difference is rounded to six decimal places. No additional bitwise
equality check was performed. Random-input correctness and untraced timings are
separate records and are not pooled with these samples.

Select `pid=2, ph=X` with exact names
`stage{1,2}_stage_ab_case_candidate_{0,1,2,3}_kernel`, sorted by integer `ts`.
There are 6 complete Stage1/Stage2 pairs for c0 and 5 each for c1/c2/c3.
The final two pairs per candidate are the measured rounds, excluding the
reference, correctness and warmup calls. Their chronological candidate order is
`0,1,2,3,3,2,1,0`, exactly matching the run log; eight `mcEventElapsedTime`
calls follow the corresponding measured pairs.

`ts`, `dur` and `submit_ts` are **integer nanoseconds**, despite the Chrome-like
trace format. Subtract epoch-sized timestamps as integers before dividing by
1000 for microseconds. Kernel end is `ts + dur`, not `args.complete_ts`.
The resulting 4.6–6.0 ms spans agree with the independently logged event windows;
interpreting `dur` as microseconds would not.

## Final two measured pairs

All duration columns below are **microseconds**. `gap=S2.start−S1.end`;
`span=S1+gap+S2`. `entry window` is the separate event-timer result from the log.

| Candidate | Round | Stage1 / Stage2 co_id | Stage1 | Stage2 | gap | span | entry window |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| v720 | 1 | 2527 / 2533 | 2929.408 | 1661.440 | 27.648 | 4618.496 | 4655.360 |
| v720 | 2 | 2884 / 2890 | 2957.056 | 1670.400 | 30.720 | 4658.176 | 4694.272 |
| v736 | 1 | 2578 / 2584 | 2962.688 | 2971.136 | 28.160 | 5961.984 | 6000.896 |
| v736 | 2 | 2833 / 2839 | 2943.232 | 2979.072 | 26.880 | 5949.184 | 5985.280 |
| v737 | 1 | 2629 / 2635 | 2922.240 | 1768.192 | 27.136 | 4717.568 | 4753.920 |
| v737 | 2 | 2782 / 2788 | 2915.584 | 1753.856 | 27.392 | 4696.832 | 4731.392 |
| v739 | 1 | 2680 / 2686 | 2942.208 | 1708.800 | 25.856 | 4676.864 | 4711.680 |
| v739 | 2 | 2731 / 2737 | 2930.176 | 1705.984 | 27.136 | 4663.296 | 4700.672 |

Arithmetic means, still microseconds:

| Candidate | Stage1 | Stage2 | gap | span | entry window | window minus span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v720 | 2943.232 | 1665.920 | 29.184 | 4638.336 | 4674.816 | 36.480 |
| v736 | 2952.960 | 2975.104 | 27.520 | 5955.584 | 5993.088 | 37.504 |
| v737 | 2918.912 | 1761.024 | 27.264 | 4707.200 | 4742.656 | 35.456 |
| v739 | 2936.192 | 1707.392 | 26.496 | 4670.080 | 4706.176 | 36.096 |

The two-sample median equals the mean here; it is not evidence of a large sample.
The window-minus-span difference is kept separate, not labeled CPU launch cost.
No other captured `pid=2, ph=X` GPU event overlaps any of these eight internal
Stage1/Stage2 gaps. All eight Stage2 submissions precede Stage1 completion by
approximately 2.91–2.96 ms. Thus delayed CPU submission of Stage2 does not explain
the gap in these pairs. This does not prove whole-device idle time or exclude
work invisible to this process's trace.

## Attribution supported by this capture

Mean differences from v720, microseconds:

| Candidate | Delta Stage1 | Delta Stage2 | Delta gap | Delta span | Delta entry window |
| --- | ---: | ---: | ---: | ---: | ---: |
| v736 | +9.728 | **+1309.184** | −1.664 | +1317.248 | +1318.272 |
| v737 | −24.320 | +95.104 | −1.920 | +68.864 | +67.840 |
| v739 | −7.040 | +41.472 | −2.688 | +31.744 | +31.360 |

v736 Stage2 duration rises **78.59%** in this capture, reproducing the large
regression in both measured rounds. Stage1 and the inter-kernel gap cannot
account for the roughly 1.3 ms loss. The independent
[generated-source audit](CODEGEN_AUDIT.md) isolated v736's C++ change to moving
one barrier; **mcTracer alone does not show why that changed Stage2's runtime**.
There are no stall counters or ISA disassembly here, so no claim of a specific
stall mechanism, changed instruction count, or duplicated physical barrier is
supported.

v737 and v739 Stage2 are also slower than v720 by 95.104 us (+5.71%) and
41.472 us (+2.49%), respectively. v739's Stage2 is 53.632 us below v737 in this
trace, but remains above the baseline in both rounds. The differences in their
unchanged Stage1 samples illustrate why whole-entry differences must not be
attributed entirely to Stage2. Two traced constant-input rounds do not establish
a small stable speedup or justify promotion over the verified baseline.

## Raw resource metadata

Every event of a given candidate/stage reports the same fields, including the
earlier excluded calls. Values below are copied from `args.mem` and the raw
occupancy-named trace fields.

| Kernel | registers_per_thread | dynamic_shared bytes | static_shared bytes | private_per_thread | private_total | mtreg_occupancy(%) raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All four Stage1 | 248 | 32768 | 0 | 0 | 0 | 48 |
| v720 Stage2 | 152 | 32768 | 0 | 0 | 0 | 29 |
| v736 Stage2 | **142** | 32768 | 0 | 0 | 0 | 27 |
| v737 Stage2 | 152 | 32768 | 0 | 0 | 0 | 29 |
| v739 Stage2 | 152 | 32768 | 0 | 0 | 0 | 29 |

All blocks are `(256,1,1)`; Stage1 grid is `(48,16,1)` and Stage2 grid is
`(48,56,1)`. Raw `shared_memeory_occupancy(%)` is 50 for every kernel.
All selected pairs use device 0, queue 0 and the same reported hardware queue.

The register count difference is observed metadata, **not** a causal
explanation. The vendor field `mtreg_occupancy(%)` is reported raw without
reinterpreting it as measured active occupancy. Zero private fields are not
proof of no spills, no extra memory traffic, or unchanged physical code.
No bandwidth/cache/stall counter values were collected in this capture.

## Artifact identity

```text
tail-1101447.json
SHA256 17ebc77ffa939d5885dcbd374f03329a240733538128f9efd77a3b3f205d0f3b
codex_profile_736_737_739_20260905.log
SHA256 29da2c12762faef0faf1da90d917892d2f087d7d336c592086a9c533f1a66607
```

Trace-only decision: reject v736's early-barrier schedule for performance;
neither v737 nor v739 shows an advantage over v720 in this capture. Keep the
constant traced results separate from random/untraced measurements and OJ.
