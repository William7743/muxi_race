# NSA062 early V in shared simple kernel

Parent057. Move ordinary T.copy(V,Vs) immediately after K copy, before QK.
No async or pipeline added. Tests changed scheduling with shared probability
and128-thread G32 PV FullCol; earlier029 used a different simple path.
Potential downside: longer Vs lifetime can prevent shared storage reuse.
No guaranteed latency overlap from source order alone.

g32062 seed413 launched, correctness/performance pending. Change currently
also affects shared-simple G16, requiring validation/gating before promotion.
NSA057 retained.
