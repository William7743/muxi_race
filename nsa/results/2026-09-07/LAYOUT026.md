# Shared-layout ablation

NSA026 uses installed TileLang make_mma_swizzle_layout vecSize4 on Qs/Ks/Vs
of original simplified S1 only. Math and chunk/online paths unchanged.
All six qk checks pass seed345, B4 L1024 H1 HQ16 S1.

|D|BS|baseline us|vec4 us|
|---|---|---|---|
|32|16|14.950|15.987|
|32|32|22.003|23.680|
|64|16|29.773|35.802|
|64|32|47.014|54.042|
|128|16|55.078|55.245|
|128|32|80.474|80.384|

D128 is unchanged control. Reject vec4. Raw layout026.jsonl/log; four ABBA
rounds ten calls. No bank-conflict counters measured, so cause remains a
hypothesis. NSA027 tests vecSize16 instead; seed346 qk running. No OJ claim.
