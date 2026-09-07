# Direct vector reduction experiments

These are local call-event measurements, not OJ results. Both probes keep
the reference paths outside S1 D<=64 unchanged. Inputs are contiguous; each
query has valid attention tokens, including partially causal current blocks.

NSA015 uses four query heads per CTA and direct FP32 multiply/reductions in
TileLang for QK and PV. It passes three configurations, but D64/BS32 fails
launch with mcErrorMemoryValueTooLarge. Partial passing results do not count
as full correctness coverage. Resource/layout pressure is a hypothesis, not
a measured register-count diagnosis.

NSA016 reduces to one head per CTA and copies probabilities to shared memory
between reductions to decouple fragment layouts. All four checks pass.

| D | block size | NSA015 baseline/probe us | NSA016 baseline/probe us |
|---|---|---|---|
|32|16|15.42 / 609.52|15.37 / 203.72|
|32|32|22.63 / 2312.38|22.04 / 672.78|
|64|16|30.35 / 6835.81|29.81 / 698.73|
|64|32|launch failure|46.82 / 2312.68|

B4 L1024 H1 HQ16 S1; seeds325/326 respectively. Four paired ABBA rounds,
ten calls per timed batch, full reference check before timing. Raw JSONL and
compilation/run logs are retained in vector015.* and vector016.*.

Reject both implementations. The shared bridge fixes this launch failure but
does not produce competitive performance. Do not promote either to submission.
