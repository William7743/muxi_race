# NSA025 ordinary safe-off CV2

Seed341 B4 L1024 H1 HQ16 D128 S1, both BS16/32 pass full reference.
BS16 exact #11580456.6144us / NSA02549.8432us. BS32 control80.0640/60.9664us.
Raw chunk025.jsonl/log. Four ABBA rounds, ten calls. BS32 uses unchanged
shared path, so its gain over #115804 does not validate the new CV2 change.

The BS16 difference versus earlier NSA024/023 sessions is too small to
interpret reliably. A direct NSA024 baseline versus NSA025 candidate run
seed342 has started in paired025. No promotion until this comparison and
broader coverage justify it. This is local event timing, not an OJ score.

Direct seed342 comparison completes both PASS. Here baseline means NSA024,
NOT #115804: BS16 51.2000us/50.0224us (ratio1.02354); unchanged BS32
61.5168/61.6960us (ratio0.99710). Raw paired025.jsonl/log. The small BS16
gain warrants wider shape testing, not immediate promotion. Full public109
against #115804 seed343 launched as public025; results pending.

Full public025 seed343: all109 PASS with exact coverage, zero missing/extra.
Overall geomean1.043981x, summed ratio1.055475x; changed33 geomean1.147975x;
unchanged76 controls1.001812x. D12831 cases1.136785x. Raw public025.jsonl/log.
Use --profile024 grouping (same changed predicate); no OJ score conversion.
Single-session wider comparison warrants repeat, not a guarantee of88 points.

Boundary validation with --include-large completes20 PASS, maximum absolute
error0.00390625. Raw edges025.log. Covers G8/G32, current/first block,
scaled Q/K and zero query plus the large shared path. Independent public109
repeat seed344 started as public025repeat; results pending.

Repeat seed344 completes all109 PASS, exact coverage. Overall geomean1.042227x,
sum ratio1.055494x. Changed33 geomean1.153085x; unchanged76 controls0.997472x.
Raw public025repeat.jsonl/log. Reproduces the first run's modest~4% gain;
not an88-point score. Next NSA026 isolates public vecSize4 shared-layout
annotations on the original simplified S1 implementation, leaving original
chunk/online paths unchanged; GPU comparison pending.
