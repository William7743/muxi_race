# Direct NSA025 versus NSA028, seed349

All10 qwide32 cases PASS for both candidates. Here baseline is NSA025, not
#115804. Raw paired028.jsonl/log. ABBA four rounds, ten calls per batch.

Changed cases, D32 H1 HQ16 S1:

|B|L|BS|NSA025 us|NSA028 us|
|---|---|---|---|---|
|4|1024|16|15.206|14.746|
|4|1024|32|21.875|20.902|
|8|1024|16|24.448|23.373|
|8|1024|32|36.954|35.840|

All four support a small3–4.5% latency reduction in the changed path.
Other six cases retain identical dispatch and serve as timing controls;
their differences are not attributed to the layout. This supports keeping
the large-grid gate, not expanding it to all shapes. Full independent repeat
and broader edge testing remain outstanding. No OJ score inferred.
