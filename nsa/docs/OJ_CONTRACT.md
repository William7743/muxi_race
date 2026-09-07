# Current NSA OJ contract

Source: user-pasted live statement, 2026-09-07.
Page: https://xpuoj.com/contest/7/problem/1

`run_kernel(q, k, v, block_indices, output, B, seq_len, H, HQ, D, S, block_size, is_causal)`

- Q/output: contiguous FP16 [B,L,HQ,D]; K/V: contiguous FP16 [B,L,H,D].
- Indices: contiguous int32 [B,L,H,S].
- Query head group g maps to KV head floor(g/(HQ/H)).
- Each selected index names a BS-token block. Sentinel index is L.
- For each selected token position p, valid means 0<=p<L and p<=t (causal).
- Compute softmax(Q K_selected^T / sqrt(D)) V_selected; write FP16 output.
- Reference computes scores and weighted sums in FP32.
- Validation is torch.allclose, atol=rtol=1e-2.
- GPU calculation must be implemented in TileLang, not PyTorch.
- No explicit synchronization in the submission is recommended.
- B: 1/2/4/8; L:64..8192; H:1/2; HQ:16/32; D:32/64/128;
  S:1/2/4/8; BS:16/32; causal:1.
- OJ lists TileLang0.1.10 and PyTorch2.8.0+metax3.7.1.5.

Unspecified: exact shape combinations, input generation, scoring/timing protocol,
whether entirely masked rows can occur, current intrinsic/pipeline policy beyond
the stated TileLang-only requirement. Do not invent these constraints or infer
them solely from the 109 public sample cases.

The public standalone example's scale=0.1 is NOT the OJ contract. Its skipped
large-case references must not be interpreted as correctness passes.
