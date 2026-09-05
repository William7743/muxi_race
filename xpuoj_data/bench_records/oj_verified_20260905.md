# OJ results verified on 2026-09-05

Read-only inspection of the user's signed-in XPUOJ pages and authenticated API.
Version mapping was checked against source headers, not guessed from submission
time or size. v714, v716, and v718 also matched local source in full after LF normalization.
All records below are Accepted.

| Source version | Submission | Display score | Case scores | Case times, ms |
| --- | --- | --- | --- | --- |
| v496 | [138992](https://xpuoj.com/contest/5/submissions/138992) | **79.67** | 81 / 79 / 79 | 2.562 / 4.749 / **9.078** |
| v515 | [139226](https://xpuoj.com/contest/5/submissions/139226) | 79.33 | 80 / 79 / 79 | 2.655 / 4.789 / approximately 9 |
| v534 | [139278](https://xpuoj.com/contest/5/submissions/139278) | 79.33 | 80 / 79 / 79 | 2.651 / 4.797 / approximately 9 |
| v691 | [139661](https://xpuoj.com/contest/5/submissions/139661) | 78.33 | 81 / 78 / 76 | 2.552 / 4.974 / **11.036** |
| v634 | [139669](https://xpuoj.com/contest/5/submissions/139669) | 78.33 | 81 / 78 / 76 | 2.584 / approximately 5 / approximately 11 |
| v713 | [139689](https://xpuoj.com/contest/5/submissions/139689) | **79.67** | 81 / 79 / 79 | 2.582 / **4.658** / 9.214 |
| v714 | [139698](https://xpuoj.com/contest/5/submissions/139698) | **80.00** | 81 / 80 / 79 | 2.594 / **4.616** / 9.207 |
| v716 | [139730](https://xpuoj.com/contest/5/submissions/139730) | **80.00** | 81 / 80 / 79 | 2.596 / 4.631 / **9.051** |
| v718 | [139753](https://xpuoj.com/contest/5/submissions/139753) | **80.33** | 81 / 80 / 80 | **2.560 / 4.600 / 8.926** |

v496, v691, and v713 case 3 were expanded to obtain exact stderr and checker values.
Other values marked approximately are rounded UI values, not precise telemetry.

Latest outcome: v718 / 139753 is the new best verified OJ score at **80.33**.
It changes only E64 Stage1 relative to v716: case 3 drops from 9.051 to 8.926 ms,
125 microseconds or approximately 1.38%, and the displayed case score increases
from 79 to 80. Total score rises from 80.00 to 80.33. E16/E32 code is unchanged;
their lower observed times in a different window are not attributed to this
E64-only change. A single score improvement does not establish repeatability.

At the preceding checkpoint, v714 and v716 tied at 80.00. v714's E32-only Stage2
change reduces case 2 by 42 microseconds relative to v713 in these two OJ runs,
crossing the displayed integer score boundary. This is not a repeatability claim.
The user completed the required browser verification and provided ID 139698;
automated read-only feedback verified the source, Accepted status, and all times.

v716 / 139730 was submitted at 2026-09-05T01:52:54.000Z. Authenticated detail
confirms Accepted, 81/80/79, with three formal cases, one sample excluded, and
zero missing results. Its source matches the repository exactly after LF
normalization; SHA256 is
`6c2e4a4d5dea28912698f8ee3c4a08004374b62dbdd34998dc182378075e4089`.
Relative to v714 it changes only E64 Stage1. Case 3 is 156 microseconds lower
(approximately 1.69%), while the total score remains 80.00. The sum of the three
reported times is 16,278 vs 16,417 microseconds. These are different submission
windows, not evidence of stable across-the-board superiority. At that checkpoint
v714 remained the 80-point reference and v716 was the faster-observed experimental
base; both are now retained as references for v718.

| v716 formal case | Time, microseconds | Measured baseline, ms | SPJ baseline, ms | SPJ score ratio | Pass |
| --- | --- | --- | --- | --- | --- |
| 1 | 2596 | 11.286 | 11.251 | 0.812523 | true |
| 2 | 4631 | 18.688 | 18.610 | 0.800740 | true |
| 3 | 9051 | 36.021 | 35.875 | 0.798535 | true |

The corresponding structured record is
[oj_139730_verified.json](v714_v715/oj_139730_verified.json).

v718 / 139753 was submitted at **2026-09-05T02:23:50.000Z**. Authenticated detail
reports Accepted, display score 80.33, and `timeUsed=16086` microseconds, equal
to the sum of its three formal case times. Full LF-normalized source equals the
repository v718; SHA256 is
`9664d1ab405354b71df30ca14bfe26b73600ecb7ac75932426e44b92c688a4d6`.
Three formal cases pass, one sample is excluded, and zero results are missing.

| v718 formal case | Display score | Time, microseconds | Measured baseline, ms | SPJ baseline, ms | SPJ score ratio | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 81 | 2560 | 11.260 | 11.251 | 0.814641 | true |
| 2 | 80 | 4600 | 18.635 | 18.610 | 0.801810 | true |
| 3 | 80 | 8926 | 35.913 | 35.875 | 0.800763 | true |

Structured record:
[oj_139753_verified.json](v717_v718/oj_139753_verified.json).

## Exact case-3 reports

v496 / 138992 stderr:

```json
{"schema_version":2,"time_ms":9.078,"speedup":3.957259,"tk_time_ms":9.078,"tb_time_ms":35.924,"pass":true}
```

Its SPJ uses baseline 35.875000 ms, user kernel 9.078000 ms, speedup 3.952x,
score ratio 0.798056, display score 79/100.

v691 / 139661 stderr:

```json
{"schema_version":2,"time_ms":11.036,"speedup":3.252537,"tk_time_ms":11.036,"tb_time_ms":35.895,"pass":true}
```

Its SPJ uses baseline 35.875000 ms, user kernel 11.036000 ms, speedup 3.251x,
score ratio 0.764746, display score 76/100.

v713 / 139689 stderr:

```json
{"schema_version":2,"time_ms":9.214,"speedup":3.909811,"tk_time_ms":9.214,"tb_time_ms":36.025,"pass":true}
```

Its SPJ uses baseline 35.875000 ms, user kernel 9.214000 ms, score ratio
0.795649, display score 79/100. The exact 9.214 ms supersedes the user's initial
rounded report of approximately 9 ms.

The case-3 reports identify experts=64, hidden=7168, intermediate=2048,
total_tokens=9088. They do not expose individual expert group sizes.

## Interpretation and limitations

- The user's initial recollection was v496=79.33; the checked v496 record is
  79.67. It was best at the first audit; v714/139698 subsequently raised the
  best verified score to 80.00. This is not an exhaustive historical audit.
- v691 is not an OJ upgrade over v496. Its case-3 path is inherited from v634;
  the E32-only final-K transformation cannot itself cause the E64 difference.
- These submissions ran at different times. They locate an important regression
  signal, not an isolated same-window causal comparison of every code change.
- Local alternating 64/220 rows are an experimental fixture. No recoverable
  official source has established that distribution as the scoring input.
- The platform evaluation guide identifies the displayed memory value as CPU
  RSS; do not infer GPU memory consumption or an input-pool size from 22.2 G.

At the v713 checkpoint, v496 remained the formal reference and v713 tied its
79.67 score. v713 isolates only v691's E32 Stage1 on top of v496. Its case-2 time is
4.658 vs 4.749 ms, approximately 1.92% lower, but this is a timing signal from
different submission windows, not an increase in the displayed case or total
score. E16/E64 code paths are unchanged; their observed time differences must
not be attributed to the E32-only transformation.

## Next isolated candidates and submission authority

On 2026-09-05 the user authorized the agent to submit through the signed-in OJ
browser and retrieve feedback, replacing the prior manual-submission-only
constraint. Any human-verification challenge is handed back to the user.

**Latest workflow supersedes that earlier authorization:** do not operate the
integrated browser. Provide code links; the user submits manually in Chrome,
then supplies an ID for the agent to retrieve feedback through read-only APIs.

- v714 starts from v713 and copies only v552's original unsplit M128 E32
  Stage2 dual-B-fragment emitter. The other shapes and all Stage1 functions
  are unchanged. It does not include the M128/M64 split.
- v715 starts from v713 and uses only v527's original unsplit E64 Stage1
  GIU/shared-merge builder. E32 retains v713's terminal-K builder and all
  Stage2 functions are unchanged.
- Source/AST equivalence, Python compilation, Ruff F/E9, and CPU mock dispatch
  for E1/E8/E16/E32/E64 passed. Each candidate exercised both route-weight
  dtypes and repeated calls with new inputs: 15 cached compiled callables and
  40 mock launches, exactly two launches per invocation. These checks are not
  GPU accuracy or performance results.
- GPU follow-up: v714 passed three random-input checks but has no stable entry
  timing gain; v715 passed three random-input checks with a 2.85% local entry
  time reduction and a separate constant-input routing retest with 0.70%.
  See `v714_v715/` raw logs and OPTIMIZATION_LOG.md for limits of these tests.
- The initial browser v714 submit returned no ID, and authenticated query
  confirmed 139689 remained the newest record at that time. A normal
  submit through the existing repository client returned HTTP 403, "Captcha
  verification failed". No bypass attempted; the user completed verification.
  Authenticated login, submission listing and detail reads are operational.
- The user then supplied 139698: v714 Accepted/80, verified above. v715 has
  no separate OJ ID. v716 combines v714's E16/E32 paths and v715's E64 path;
  source/AST/CPU dispatch checks passed and the composition now independently
  passed OJ as 139730, Accepted/80.
- v717 adds only E64 Stage2 dual-B emitter to v716 and passed local random and
  separate constant-input checks; it has no OJ ID. v718 adds only E64 Stage1
  terminal-K, passed its local checks, and is now independently Accepted/80.33
  as 139753. The two changes were not stacked in v718.
- The next candidate is v719, based on v718 with only E64 Stage2 selecting the
  existing dual-B emitter. All builder bodies are unchanged. Python/Ruff,
  source/AST, and CPU mock checks passed. Three NaN-poisoned recomputations of
  the same random input batch and two recomputations of a separate constant-input
  fixture matched v718 exactly for the full chain and real entry. Entry time was
  approximately 0.60%/0.48% lower, with all four pairs lower in each fixture.
  The constant fixture is not random-input validation; these small local gains
  do not guarantee an OJ score increase. GPU testing ended and the lock was
  released. v719 is ready to provide as a code link for user-operated Chrome
  submission; it has no OJ ID or score yet. Logs:
  [same-batch random](v719/codex_e64_718_719_entry_random.log),
  [synthetic constant](v719/codex_e64_718_719_entry_synthetic_constant.log).
