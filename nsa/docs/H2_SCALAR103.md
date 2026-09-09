# NSA103: preserve NSA097 and specialize small H2 output conversion

Candidate: `probes/probe_nsa103_h2_small_scalar_output.py`.
Normalized LF SHA256: `6115cd1595aaa26f518e362810ab50771ade4812b0b35fb8d81444d471f5795a`.
This is a local candidate, not an accepted OJ submission or an88-point result.

## Why this version

NSA099 was not worthwhile as an all-D64 change, but its H2 small shape showed
a reproducible improvement. A reverse-order independent run with seed137,
public/current block selection and five rounds confirmed the case13 benefit:

| Case / mode | NSA097 us | NSA099 us | Guard-relative elapsed change |
|---|---:|---:|---:|
| 13 / public | 8.00768 | 7.71072 | -4.45% |
| 13 / current | 7.98720 | 7.74144 | -3.79% |
| 4 / public | 10.56768 | 10.55232 | +0.58% |
| 4 / current | 10.58816 | 10.54208 | -0.29% |
| 14 / public | 16.29184 | 16.22528 | -0.56% |
| 14 / current | 16.33280 | 16.26112 | -0.61% |

All12 paired reference checks and60 timing samples passed the declared gates.
Cases4/14 do not justify extending the specialization; small improvements below
1% are not treated as established optimizations. Raw data and summaries are
`nsa099_h2_confirm.jsonl`, its `.log`, and `nsa099_h2_summary.json`.

## Implementation and isolation

Starting from NSA097, add one direct-JIT factory copied exactly from NSA099's
validated `nsa_simplified` body. Select it only when the previous selection is
`nsa_simplified`, H=2, D=64, G=16, BS=16, and B*L*H<=1024. S1 is implicit in
the previous factory selection. These are ordinary shape-based choices, not
input/result caches, invocation counts, or correctness/benchmark distinctions.

The factory copies16 FP32 accumulator values into per-lane local storage,
converts them individually to FP16, writes the established MMA coordinates to
shared memory, and retains the original wide global output store. QK, softmax,
PV, normalization, bounds, offsets, and synchronization are unchanged. Its
body/decorators exactly match the already GPU-tested NSA099 factory; the audit
also proves every original NSA097 function unchanged except this dispatch gate.
No new async/pipeline, external code, or custom compiler flags are introduced.

## Independent final-file check

The final NSA103 file was tested against NSA097 with a third seed42/public,
all14 points, three alternating rounds. All28 reference checks and84 timing
samples passed. Case13: NSA0977.99744us -> NSA1037.82336us, raw elapsed-time
reduction2.18%; guard-relative reduction2.64%. This is a small one-point gain,
not a whole-operator2.64% gain. Whole-suite geometric relative time0.998413
includes noise on unchanged kernels and is not presented as a general speedup.

Raw data: `nsa103_full.jsonl`, `nsa103_full.log`, `nsa103_summary.json`.
No OJ score is inferred from these local16GB measurements. The last verified
accepted anchor remains v236/#14159485.93; NSA097 and NSA103 need OJ validation.

Additional regression completed:8 shapes x2 input updates x2 implementations
=32/32 reference assertions passed (16 for NSA103). Shapes include B1 with
L64/128/256/257/512/513 and B2 with L256/257, all H2/HQ32/D64/S1/BS16.
This crosses the1024-query-work threshold and checks non-aligned last blocks.
The second update changes Q/K/V in the same storage (Q/K x4, V x16) and changes
indices to each query's current block. Candidate total:14+16=30 checks; paired
total:60. These counts are assertions, not60 distinct shapes. All outputs
were finite and matched the FP32 reference with atol=rtol=0.01.

`nsa103_verification.json` binds these results to source hashes and verifies
the13 unchanged generated CUDA fingerprints. `sources103/` includes case13
parent/candidate output code. This is generated-source equality, not a claim
that full device binary objects were compared.

## Verification commands and limitations

Locally run `python nsa/tests/audit_nsa103.py` for AST/source isolation.
On the configured C500 environment, run the pinned external calibration script
with `--modules submissions/nsa097.py submissions/nsa103.py --seeds 42
--modes public --profiles cupti --rounds 3 --guard-module submissions/nsa_v159.py
--output results/nsa103_full.jsonl`.
Use `tests/stress_nsa_h2_small.py` for H2 branch boundaries and in-place updates.
Use `tests/verify_nsa103_results.py` to check archived source identities,
coverage, pass/fail counts and unchanged generated-source fingerprints.

The server staging layout uses `tests/benchmark.py`, `tests/oj_cases.json`,
`tests/calibrate_oj_protocol.py` and the guard reference JSONL from external
William7743/NSA commit bea81966fe81e0a2e9fcdd5991b6efac78cdcf7c.
Copy the two candidate files to `submissions/nsa097.py` and `submissions/nsa103.py`
there; copy the new stress script next to `benchmark.py`. The stress invocation
is `python tests/stress_nsa_h2_small.py --modules submissions/nsa097.py
submissions/nsa103.py --output results/nsa103_stress.jsonl`.
Use fresh output names for a new run; the committed logs are immutable evidence.

The local setup is the user-provided C50016GB sGPU with25% compute; results
are not measurements of the unavailable full64GB OJ GPU. The static module
assignment audit is not the OJ SafeExecutor. All-masked rows and noncausal
inputs are not validated by the added regression suite.
