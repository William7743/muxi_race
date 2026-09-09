# 16 GB instance: external NSA baseline reproduction and NSA096

## Provenance and scope

External source: [William7743/NSA at bea81966](https://github.com/William7743/NSA/tree/bea81966fe81e0a2e9fcdd5991b6efac78cdcf7c).
The existing server directory was not modified. Experiments ran in a separate
directory, using copied source and the snapshot's testing scripts.
No OJ submission made and no credentials included in this report.

The external OJ export records v159/#141137=85.79, v176/#141138=84.93;
v318 has no OJ result in that snapshot. All 19 exported submissions are
Accepted. The user's additional failing candidates have not yet been identified.
The export resolves earlier score gaps: #141056=67.71 and #141060=84.71.
These are imported records, not newly fetched platform responses.

Machine: C500 sliced GPU, compute quota 25%, VRAM quota 16000 MiB,
MACA 3.7.1.5, driver 3.8.30. GPU process list was empty before tests.
The physical board's 65536 MiB display is not this instance's usable quota.

v318 LF-normalized source SHA256:
`ce6aed759f07749dae186852d9781277feed4626c667ffc62e9a0d0141f6b383`.
Windows checkout/upload has CRLF and a different raw SHA; the raw bytes used
for measurements are recorded in each JSONL row. LF normalization explains
the difference from the original server file; no mathematical edit is implied.

## Reproduction protocol

Use the snapshot's `tests/calibrate_oj_protocol.py`, `benchmark.py`,
`oj_cases.json`, and `initial_profiler_paired.jsonl` reference. Kernel inputs are
fresh contiguous FP16 tensors and int32 indices; reference math is FP32.
Seed=0/public, three alternating rounds, common output address, v159 control
before and after each measurement. cupti uses 10 warmups, 50 repeats and the
existing 256,000,000-byte cache-flush protocol. No per-point timing-method selection.

Keep samples only for analysis when control drift <=3% and reference ratio
is within [0.92,1.08]; this is not a confidence interval for tiny improvements.
Raw samples remain in the files. 82/84 samples satisfied both conditions;
every module/point retained at least two. 28/28 reference checks passed.

| Point | v159 local us | v318 local us |
|---|---:|---:|
| 1 | 4.910 | 5.018 |
| 2 | 5.970 | 5.970 |
| 3 | 8.714 | 8.740 |
| 4 | 10.609 | 10.542 |
| 5 | 25.687 | 25.390 |
| 6 | 89.723 | 81.659 |
| 7 | 24.776 | 24.750 |
| 8 | 41.631 | 40.750 |
| 9 | 40.453 | 40.458 |
| 10 | 7.224 | 7.286 |
| 11 | 19.026 | 19.103 |
| 12 | 71.086 | 62.904 |
| 13 | 7.936 | 7.977 |
| 14 | 16.420 | 16.476 |

Main gains reproduce at points6/12 (~9%/~11.5% elapsed-time reduction).
Other differences are small and do not establish reliable per-point gains.
No new OJ score is inferred from this table. Hidden values, system scheduling
and scoring-rounding differences remain unverified.

Raw records: `results/2026-09-09/nsa_repo_recheck.jsonl` and accompanying log.

## NSA096: D64 probability conversion

Self-contained experimental source: `probes/probe_nsa096_v318_d64_scalarprob.py`.
Parent is exactly external v318 above. Four S1 factories replace only the
FP32->FP16 probability copy, guarded by D64/G16/BS16. The scalar conversion
pattern is borrowed from the parent's S8 path. Input dispatch, mathematics,
loads, masks and synchronization are unchanged at the TileLang source level.
No new compiler options or pipeline mechanisms added; inherited settings remain.

`tests/audit_nsa096.py` restores the four replacements and requires all other
AST nodes to equal v318, including decorators and dispatcher. Syntax and this
bounded source isolation check passed. It is not a general layout/race proof.
Candidate LF SHA: `a5f2e447a90f009b4c1c628968ecf2348311577b4b50e38211bde3425d7a2e62`.

Initial points2/4/5/7/8/9/13/14, three rounds: 16/16 paired reference checks,
48/48 timing samples passed drift/reference filters. Points4/5 appeared about
1% faster; others near parity. Raw file `results/2026-09-09/nsa096.jsonl`.
Reverse-load seed137 public/current confirmation completed: 12/12 reference
checks, 60/60 timing samples passed the same filters. Point4 was near parity;
point5 regressed about1%; point8 retained only~0.5%-0.8% lower relative time.
The initial apparent overall benefit did not reproduce. NSA096 is not promoted
and is not recommended for OJ. Keep v318; no claim of progress to88 points.

| Confirmation | NSA096 local us | v318 local us |
|---|---:|---:|
| point4 public | 10.557 | 10.583 |
| point4 current | 10.583 | 10.578 |
| point5 public | 25.830 | 25.559 |
| point5 current | 25.754 | 25.569 |
| point8 public | 40.361 | 40.550 |
| point8 current | 40.428 | 40.678 |

Raw confirmation: `results/2026-09-09/nsa096_confirm.jsonl` and log.
Generated sources for points2/4/8/9 are retained in `sources096/` under that
results directory. Manual inspection of point8 confirms four original
probability registers are copied in order into the scalar conversions and
back into the four acc_cast registers. No output-cast change was introduced.
This bounded inspection is not a general numerical or race-freedom proof.
Because performance did not qualify, wider release regression is not claimed.

## Reproduce

On C500, from the pinned external snapshot with the candidate copied into
`submissions/nsa096.py`, load `tests/env_c500.sh` (use LF shell files):

```sh
python tests/calibrate_oj_protocol.py --modules submissions/nsa_v159.py submissions/nsa_v318.py --seeds 0 --modes public --profiles cupti --rounds 3 --guard-module submissions/nsa_v159.py --output results/recheck.jsonl
python tests/calibrate_oj_protocol.py --modules submissions/nsa_v318.py submissions/nsa096.py --ids 2 4 5 7 8 9 13 14 --seeds 0 --modes public --profiles cupti --rounds 3 --guard-module submissions/nsa_v159.py --output results/nsa096.jsonl
python tests/calibrate_oj_protocol.py --modules submissions/nsa096.py submissions/nsa_v318.py --ids 4 5 8 --seeds 137 --modes public current --profiles cupti --rounds 5 --guard-module submissions/nsa_v159.py --output results/nsa096_confirm.jsonl
```

Offline source/timing audit from muxi repository root:

```sh
python nsa/tests/audit_nsa096.py --baseline PATH_TO_PINNED_NSA/submissions/nsa_v318.py --candidate nsa/probes/probe_nsa096_v318_d64_scalarprob.py --measurements nsa/results/2026-09-09/nsa096.jsonl
```
