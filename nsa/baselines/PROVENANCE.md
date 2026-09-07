# Baseline provenance

## OJ #115804

- User supplied the complete source on 2026-09-07 and identified submission
  **#115804**, with a previously reported score of **82.86**.
- Frozen source: `oj_115804.py`.
- SHA256 of archived file:
  `63d9d2dcd4ad77adc123718fdebddf1384d1e68b4c723f217571e59db2ccaac0`.
- Submission identity and score are user-reported, not independently retrieved
  from OJ during this session. No new OJ submission was made.
- Compared with `nsa_20260817.py`, the only source difference is the host-side
  `block_indices.dtype != torch.int32` guard and conversion in `run_kernel`.
  The TileLang kernel definitions and dispatch are identical. For the published
  contiguous int32 input contract, no conversion executes.
- Existing local correctness evidence for `nsa_20260817.py` covers identical
  device computation, but should not be relabeled as a fresh execution of the
  exact #115804 entry point or a reproduction of its OJ score.
- Historical performance/correctness claims in frozen source comments are
  preserved for provenance, not adopted as newly verified conclusions.
