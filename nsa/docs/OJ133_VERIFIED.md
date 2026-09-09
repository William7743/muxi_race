# NSA133 / OJ141741: verified86.64, no packed-V benefit established

Direct authenticated **read-only** query confirms Accepted86.64 on2026-09-09.
The downloaded source matches probe_nsa133_d128_packed_v_control.py except outer
whitespace; complete AST including docstrings matches. No downloaded code was
executed. No submission was made by this task.

Submitted LF SHA256:
`4eb61ece5a3783050439989aa23ed6d058c92f7d64523ea9db404ad2a2c65f2b`.
Frozen probe SHA256:
`6f87a1d3a99038c5e5d6fc99f220abd0d9bd7f8373c71bc37bb416d3d27b0a15`.

## Attribution against128 /141726

| Point | Generated source changed? |128 OJ|133 OJ|
|---|---|---|---|
|6: D128/BS32 target|Yes|85us /84|85us /84|
|12: S8|No|61us /83|62us /83|
|13: H2 small|No|9us /89|8us /90|

All other reported checker times and scores match. Scores total1213 versus1212;
88 requires1232 integer point-score units, leaving19. The0.07 displayed increase
is entirely on an unchanged kernel path. This does not establish packed-V gain.
It is consistent with timing/context variability, but its cause is not measured.
Rounded checker times and one submission per version cannot establish a causal
hardware-speed comparison. Generated CUDA equality is local evidence, not proof
that OJ compiled binaries are byte-identical.

## Local full regression

Own guarded workflow183779/183780 completed in64.48s, exit0, no guard yield.
Hardware was observed idle before launch: C50016GB quota,25%compute sGPU,
MACA3.7.1.5. The host OOM counter remained9. This is not a full64GB C500.

Seed563, public/current index modes, all14 formal shapes, two versions:
56/56 full FP32-reference checks passed.112 raw timing samples,111 qualify using
unchanged guard gates (<=3% drift and reference ratio0.92–1.08). The133 point13
current-mode row has only one qualified timing and is excluded from comparison;
it was not rerun merely to obtain a favorable qualified result.

Target point6 normalized133/128 ratios: public1.0116968/current1.0116773,
about1.17% slower, consistent with the earlier ~1.39% target regression.
All28 exported sources bind to recorded code hashes; only point6 differs.
The previous AST audit also restricts133 changes to packed first-V storage.

## Decision and reproduction

Preserve133 as the **actual highest-scoring submitted artifact**.
Keep128 as the engineering control; do not incorporate packed-V into future
candidates solely because133's total score is higher. NSA134 is still an
unsubmitted S4 synchronization candidate; its OJ score is unknown.
Next gains must be supported on the changed path and beat a real point threshold,
not be selected by unchanged-path total-score fluctuations.

From repository root:

```bash
python nsa/tests/audit_nsa132_133.py
python nsa/tests/verify_oj133.py
```

The verifier reads sanitized OJ details and source-whitespace binding offline
when the ignored local capture is absent. It checks score arithmetic, source
identity, full reference coverage, timing exclusions and generated-code changes.
Status PASS covers those checks; it does not label all112 timing samples valid.
GPU reproduction driver: nsa/tests/run_nsa133_validation.py under the exclusive
supervisor with the existing pinned benchmark helpers.

Evidence in results/2026-09-09: oj133_verified_details.json,
oj133_source_binding.json, oj133_verification.json, nsa133_full.jsonl,
nsa133_full_guarded.{json,log}, sources133/*.cu.
Raw credentials/authentication responses stay excluded. No peer work was stopped.
