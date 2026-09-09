# NSA142 combination and S8 address controls143/144

## Decision

NSA142 is an optional manual OJ candidate combining the independently tested
H2 bounded-load138 and S4 explicit-sync134 paths. It does not include rejected
D128 packed-V133/combined-output141 or the new S8 experiments.
No new OJ result exists; actual highest remainsNSA133/#141741/86.64.
Goal88 is not met. Raw measurements and earlier counterexamples remain preserved.

142 SHA256:b877d4b3a93e3c330bc0c47b7dd41ba7536453a8b9ed012f452b908d95a4708a.
143 SHA256:d6a8f0233d04fdcd7a84e5df64e9d848d3a5f4d974db3615dcaee36561f950b2.
144 SHA256:df995ed2d8554688455823655f4143501570fd3cb4f28eebe9e9b0a164f036d6.

## Composition audit

Start with138, copy the complete nsa_gather_sync134 factory byte-equivalently
at AST level, and add134's existing S4/D64/G16/BS16 dispatch condition.
H2/S1 and S4 gates are disjoint. All other functions and dispatch conditions
match138. Full generated sources confirm only points11/13 differ versus128;
point11 equals the measured134 donor, point13 equals the measured138 donor.

Initial text extraction accidentally included a duplicate _get_kernel because
the donor returns gather_kernel rather than kernel. The static audit rejected
it before any GPU run. It was corrected and reuploaded; final source hashes
were checked on the server before launch. The audit now explicitly rejects
duplicate top-level function names. Only corrected files were GPU-tested.

## Full142 validation

Guarded workflow189406/189407 exited0 in101.80s, no yield. Device was idle before
launch, C50016GB/25%compute sGPU, MACA3.7.1.5; not a full64GB OJ-equivalent device.
No other task was stopped; host OOM counter stayed9. All own jobs are terminal.

Seed601, public/current modes, two rounds per module/point:
56/56 full FP32-reference assertions,112/112 qualified timing samples.
The fixed guard gates remain<=3% before/after drift and reference ratio0.92–1.08.

| Changed point |142/128 public time|142/128 current time|
|---|---:|---:|
|11, S4|0.98497|0.98414|
|13, H2|0.97686|0.99699|

Point11 improves about1.5%. Point13 varies between2.3% and0.3%, weaker in one
mode than standalone138's earlier~2% results. Generated donor code is identical;
do not advertise a uniform or additive speedup or infer OJ score from these runs.

Additional full-output correctness:32 H2 assertions (eight B/L combinations,
two fresh in-place updates, two versions) and48 S4 assertions (eight shapes,
three fresh updates, two versions), all pass with finite outputs.
They include aligned/nonaligned lengths, partial current blocks, amplified Q/K/V,
and S4 BS32/D128 fallbacks. This is finite coverage, not proof for every input.
All28 full-suite source captures bind to the tested Python source hashes.

## S8 isolated controls

143/144 start from142 and change only nsa_online_direct_output addressing.
The existing dispatch already guarantees S8/H1/G16/D64/BS16 and aligned L>=1024.
For a admitted causal nonnegative block,0<=block_id<=t//BS implies block_id<L//BS.
Hence the parent's clamp is an identity and modulo does not wrap a sentinel.
143 checks the original block ID and uses(block_id%(L//BS))*BS for bounded loads.
144 uses the same predicate but directly uses block_id*BS. Attention arithmetic,
mask positions, packed prefetch, loops, tiles and synchronization are unchanged.

Seed599 case12, two modes, three rounds:6 reference assertions and18/18 qualified
timings pass.143 ratios1.00051/1.00144 are flat/slightly slower;144 ratios
1.03777/1.04117 are about4% slower. Neither is promoted or fully certified.
144's exported K/V pointers use64-bit expressions while142's bounded expressions
use32-bit arithmetic. This is compiler evidence consistent with extra addressing
work, not a measured proof of the slowdown's unique cause.143's new bound form
does not improve on the parent's already-bounded clamp.

## Reproduction

From repository root:

```bash
python nsa/tests/audit_nsa142_144.py
python nsa/tests/verify_nsa142_144.py
```

GPU driver:run_nsa142_144_validation.py under the existing exclusive supervisor
with pinned staged benchmark/reference helpers. Target artifacts, full sources,
raw JSONL, stress rows and supervisor logs are under results/2026-09-09:
nsa143_144_screen, artifacts143_144, nsa142_full, sources142,
nsa142_h2_stress, nsa142_s4_stress, nsa142_144_guarded and verification.
Only142 is suggested for optional manual OJ testing; no automatic submission.
