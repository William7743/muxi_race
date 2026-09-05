# OJ results verified on 2026-09-05

Read-only inspection of the user's signed-in XPUOJ submission pages. Version
mapping was checked against the visible submitted-source header, not guessed
from submission time or code size. All records below are Accepted.

| Source version | Submission | Display score | Case scores | Case times, ms |
| --- | --- | --- | --- | --- |
| v496 | [138992](https://xpuoj.com/contest/5/submissions/138992) | **79.67** | 81 / 79 / 79 | 2.562 / 4.749 / **9.078** |
| v515 | [139226](https://xpuoj.com/contest/5/submissions/139226) | 79.33 | 80 / 79 / 79 | 2.655 / 4.789 / approximately 9 |
| v534 | [139278](https://xpuoj.com/contest/5/submissions/139278) | 79.33 | 80 / 79 / 79 | 2.651 / 4.797 / approximately 9 |
| v691 | [139661](https://xpuoj.com/contest/5/submissions/139661) | 78.33 | 81 / 78 / 76 | 2.552 / 4.974 / **11.036** |
| v634 | [139669](https://xpuoj.com/contest/5/submissions/139669) | 78.33 | 81 / 78 / 76 | 2.584 / approximately 5 / approximately 11 |

v496 and v691 case 3 were expanded to obtain exact stderr and checker values.
Other values marked approximately are rounded UI values, not precise telemetry.

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

Both reports identify experts=64, hidden=7168, intermediate=2048,
total_tokens=9088. They do not expose individual expert group sizes.

## Interpretation and limitations

- The user's initial recollection was v496=79.33; the checked v496 record is
  79.67. This is the best score verified in this audit, not an exhaustive audit
  of every historical submission.
- v691 is not an OJ upgrade over v496. Its case-3 path is inherited from v634;
  the E32-only final-K transformation cannot itself cause the E64 difference.
- These submissions ran at different times. They locate an important regression
  signal, not an isolated same-window causal comparison of every code change.
- Local alternating 64/220 rows are an experimental fixture. No recoverable
  official source has established that distribution as the scoring input.
- The platform evaluation guide identifies the displayed memory value as CPU
  RSS; do not infer GPU memory consumption or an input-pool size from 22.2 G.

The formal baseline remains v496. v713 isolates only v691's E32 Stage1 on top
of v496; it must obtain its own OJ result before replacing the baseline.
