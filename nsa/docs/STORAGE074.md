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
