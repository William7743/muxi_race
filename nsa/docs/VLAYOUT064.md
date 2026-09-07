# NSA064 V-only vec4 layout

Parent057. Annotate only Vs with make_mma_swizzle_layout vecSize4 in
shared simple kernel. Generated057 PV accesses Vs via per-element source
loads; alternative layout may help or hurt. Source syntax does not prove
machine load width or bank conflicts. Q/K/prob layouts, threads, math,
load timing unchanged. Initial g32064 seed415 launched; not promoted.
G16 also affected by this experimental source, needs gating/test if useful.

g32064 all7 PASS. Changed candidate20.147,20.339,34.445,60.378,
115.354us: slower than recent057, reject064. NSA065 changes only the
Vs annotation to vec16; g32065 seed416 launched to test wider layout.
