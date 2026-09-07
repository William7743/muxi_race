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

NSA027 seed346 completes all6 PASS. Baseline/probe microseconds:
D32 BS16 15.334/14.733, BS32 22.131/21.478;
D64 BS16 29.683/31.130, BS32 46.746/45.581;
D128 controls BS16 55.398/55.475, BS32 80.115/80.346.
Raw layout027.jsonl/log. No global promotion. D32 improvements need broader
validation: added qwide32 suite with five B/L configurations and both block
sizes (ten cases), seed347 launched. No broader gain established yet.

Widening completes all10 PASS, raw wide027.jsonl/log. Small cases mixed;
B4L1024 BS16 14.976/14.810us, BS32 21.760/20.915;
B8L1024 BS16 24.896/23.488, BS32 36.890/35.699.
NSA028 gates the new layout to S1 D32 B*L*H>=4096 on parent NSA025.
Other paths retain parent dispatch, G8 fallback has priority. Full public109
seed348 launched, no full-suite or OJ gain claimed yet.
