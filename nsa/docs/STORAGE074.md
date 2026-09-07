# NSA074 storage rewrite isolation

Installed pass_config.py defines TIR_DISABLE_STORAGE_REWRITE as
tirx.disable_storage_rewrite. Duplicate original057 simple JIT with this
flag enabled only for D64/S1/BS16/groups>=16 dispatch. All mathematics,
tiles and thread count unchanged. Thread synchronization pass stays enabled.
Hypothesis: distinct shared temporaries may reduce alias-related barriers;
extra shared memory may reduce occupancy. Neither outcome assumed.

Syntax PASS. d64074 seed425 d64bs16 suite launched; results pending.
NSA057 retained. Source inspection and local measurements do not prove OJ88.

Completed exact29/29 PASS. Reference/candidate geomean0.990849370x,
cumulative ratio0.996748124x. No meaningful gain; not promoted. Raw log
and JSONL retained. This does not establish that allocation/barriers changed;
only the effect of this pass-config experiment was measured.

Follow-up source074 export completed for B4/L1024/H1/HQ16/D64/S1/BS16.
Generated source is byte-identical to source057simple_d64.txt, SHA256
240e8910af47d09b6c81330cd897a67aa7b495ffee6138bb9b46627a692e1335.
Thus this configuration did not change allocation/barriers at all. Installed
engine/phase.py line256 directly invokes tilelang.transform.StorageRewrite.
Do not treat074 timing as evidence against distinct shared allocation in
general. Before another compiler-flag benchmark, verify source changes first.
