# NSA021 shared K/V reuse

Seed332, B4 L1024 H1 HQ16 S1, all six correctness checks pass for both
baseline and candidate. Only the simplified S1 path changes; D128 is an
unchanged chunk-path control. Math and probability normalization unchanged.

|D|BS|baseline us|candidate us|
|---|---|---|---|
|32|16|16.269|15.526|
|32|32|21.773|22.144|
|64|16|29.709|30.003|
|64|32|47.117|45.683|
|128|16|55.757|55.706|
|128|32|80.973|81.037|

ABBA four rounds, ten calls per batch. Raw reuse021.jsonl/log. No consistent
improvement; do not merge. Source-level shared buffer elimination does not
prove lower generated shared usage; compiler may already reuse storage.
tools/export_kernel_source.py exports actual TileLang-generated source for
inspection only. It does not bypass TileLang or execute external kernels.

Compiler export completed for D32/D64 BS16, files source021/base*.txt and
probe*.txt. D64 baseline already aliases Qs and Vs at shared offset0;
Ks is at2048. NSA021 simply moves V accesses to Ks at2048 and removes the
Vs pointer declaration. The generated data buffers still occupy the same
two regions; explicit reuse did not shrink that footprint. This explains
why source-level allocation counting was misleading. No occupancy gain
is established. These files are compiler output only, not hand-written
external device implementations to be submitted or executed separately.
