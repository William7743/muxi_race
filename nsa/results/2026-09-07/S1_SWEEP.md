# S1 D128 chunk sweep

`tests/sweep_s1.py` changes only CK and CV constants in the exact #115804
source, in memory. Generated variants are experimental test artifacts, not
standalone submissions. CK splits QK's reduction dimension; CV splits PV's
output dimension. No result caching or input-dependent phase dispatch.

Two synthetic supported configurations (not asserted to be exact OJ cases):
B4/L1024/H1/HQ16/D128/S1/BS16 and B2/L2048/H1/HQ16/D128/S1/BS32.
Fresh normal inputs, seed2026; random indices include the current partial
block, exercising causal masks beyond fully historical blocks. Entire output
is compared with FP32 reference cast to FP16, atol=rtol=0.01.

Four alternating forward/reverse orders; one warmup then ten direct callable
invocations per event measurement. These timings exclude run_kernel host
dispatch but may include host launch gaps. They are not OJ scores.

First sweep: all eight variant/case checks PASS.

| CK/CV | BS16 median us | BS32 median us |
|---|---:|---:|
|2/4 (baseline)|55.04|80.04|
|1/2|68.88|114.38|
|2/2|57.06|81.82|
|1/4|64.44|116.25|

No tested coarser split improved on baseline. Raw rounds are retained in
`sweep_s1_2026.jsonl`. Finer-split sweep is a separate experiment.

Finer sweep: all eight checks PASS, same inputs/protocol.

| CK/CV | BS16 median us | BS32 median us |
|---|---:|---:|
|2/4 (baseline)|54.60|79.97|
|4/4|74.21|69.20|
|2/8|158.18|112.97|
|4/8|487.77|284.79|

CK4/CV4 lowers BS32 latency by approximately 13.5% in this experiment,
but regresses BS16. This supports investigating shape-based selection,
not replacing the baseline globally. Need new-seed repeat, more shapes,
and run_kernel end-to-end checks before promoting it. CV8 is strongly rejected
for these cases. Raw rounds: `sweep_s1_fine.jsonl`.

New-process seed2027 repeat: BS32 baseline80.96 us vs CK4/CV4 68.67 us,
approximately15.2% latency reduction, full correctness PASS. Both shapes and
variants passed again; raw rounds in `sweep_s1_repeat.jsonl`.

`probes/probe_nsa003_bs32_ck4.py` freezes the proposed shape rule: only the
existing D128 chunk path with BS32 uses CK4, all other paths unchanged.
This file passes Python syntax checking; full entry-point testing and broader
shape coverage are still pending. Do not submit based only on this sweep.
