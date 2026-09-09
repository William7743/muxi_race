# NSA128: OJ141726 directly verified

Read-only authenticated detail read on2026-09-09 confirms Accepted,86.57,
all14 formal scoring results. No OJ submission was made by this workflow.
The earlier user-report-only record remains historical; direct safe details
are now archived as oj128_verified_details.json.

Submitted LF SHA:6c111f0bcc8b6e8ba32c0fd23c16a3f92076ca7162a7241b85afd7e608bdaee0.
Published128 SHA:309e71c21ea3190d38b2ba9ef82cc295bb61b66bebf4aa332c6f757ba83db57a.
The only difference is trailing whitespace/newline removal. Stripped source
text and the complete Python AST, including docstrings, match exactly.
oj128_source_binding.json permits reconstruction and offline hash verification
without republishing private raw API responses or a second source copy.

## Checker comparison

These are checker-message microsecond values, not alternate telemetry or local
times. Checker baseline times and shapes match the earlier109/#141658 anchor.

| Point |109 time us|128 time us|109 score|128 score|Kernel changed locally?|
|---|---|---|---|---|---|
|1|5|5|87|87|No|
|2|6|6|88|88|No|
|3|9|9|88|88|No|
|4|12|11|87|88|Yes|
|5|29|26|82|83|Yes|
|6|85|85|84|84|No|
|7|28|25|88|89|Yes|
|8|44|43|84|85|No|
|9|44|42|83|84|Yes|
|10|8|7|88|90|No|
|11|21|21|86|86|No|
|12|65|61|82|83|Yes|
|13|9|9|89|89|No|
|14|17|17|88|88|Yes|

Generated-code differences are reconstructed from source-bound full datasets
109->123->127->128, not inferred from version labels. Five of the eight gained
integer point-score units occur on changed paths; three occur on unchanged
paths. This is not a causal split of hardware/code effects. Do not attribute
the whole0.57 gain to D64 bounds/S8 packing or fit a correction from two totals.

The integer point-score sum is1212 (mean86.571428...).88 requires1232, leaving20
units. Thresholds for each next1/2/3 point score are saved in oj128_verification.json.
For example, point6 needs <=81.176us for85, point11 <=19.874us for87, and
point12 <=58.095us for84. Timing rounding/noise can affect boundary outcomes.

## Reproduce the CPU audit

```text
python nsa/tests/verify_oj128.py
```

It checks Accepted/status/source hashes/AST, all14 checker formulas, shape and
baseline invariance, full-run source bindings and changed/unchanged-path deltas.
When the private download is absent it reconstructs source solely from the
published probe plus recorded outer whitespace and checks the fetched source hash.
API login used the normal existing challenge flow; credentials were entered
through hidden input and were not written to the repository. Raw detail data
and downloaded source remain in ignored.local storage. Local test hardware is
still C50016GB/25% slice, not a newly acquired full64GB device.88 is not achieved.
