"""Independent local smoke test; PyTorch is reference/timing only, not submission.

Matches the user-provided current OJ token-level validity rule. Timings are
exploratory (one warmup, one timed invocation), not final performance claims.
"""
import importlib.util
import json
import sys
import time
import torch


def reference(q, k, v, indices, bs, causal):
    b, length, hq, dim = q.shape
    h = k.shape[2]
    groups = hq // h
    starts = indices.long() * bs
    positions = (starts[..., None] + torch.arange(bs, device=q.device)).flatten(-2)
    rows = torch.arange(length, device=q.device)[None, :, None, None]
    valid = (positions >= 0) & (positions < length)
    if causal:
        valid = valid & (positions <= rows)
    pos = positions.clamp(0, length - 1)
    bi = torch.arange(b, device=q.device)[:, None, None, None]
    hi = torch.arange(h, device=q.device)[None, None, :, None]
    keys = k[bi, pos, hi].float()
    vals = v[bi, pos, hi].float()
    query = q.reshape(b, length, h, groups, dim).float()
    scores = torch.einsum('blhgd,blhnd->blhgn', query, keys) * dim ** -0.5
    scores.masked_fill_(~valid[..., None, :], float('-inf'))
    probs = scores.softmax(-1)
    return torch.einsum('blhgn,blhnd->blhgd', probs, vals).reshape_as(q)


def main():
    spec = importlib.util.spec_from_file_location('candidate', sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = [(1, 64, 1, 16, 32, 1, 16, True),
             (1, 128, 1, 16, 128, 1, 32, True),
             (1, 128, 2, 32, 64, 4, 16, True)]
    if len(sys.argv) > 2:
        cases = [cases[int(sys.argv[2])]]
    for case in cases:
        b, length, h, hq, dim, selected, bs, causal = case
        for seed in (0, 17):
            torch.manual_seed(seed)
            q = torch.randn(b, length, hq, dim, device='cuda', dtype=torch.float16)
            k = torch.randn(b, length, h, dim, device='cuda', dtype=torch.float16)
            v = torch.randn_like(k)
            idx = torch.full((b, length, h, selected), length, dtype=torch.int32)
            for ib in range(b):
                for t in range(length):
                    for ih in range(h):
                        choices = torch.randperm(max(1, t // bs))[:selected]
                        idx[ib, t, ih, :choices.numel()] = choices
            idx = idx.sort(-1).values.cuda().contiguous()
            output = torch.full_like(q, float('nan'))
            assert all(x.is_contiguous() for x in (q, k, v, idx, output))
            print(json.dumps(dict(case=case, seed=seed, phase='before_run')), flush=True)
            started = time.perf_counter()
            mod.run_kernel(q, k, v, idx, output, *case)
            print('run_returned', flush=True)
            torch.cuda.synchronize()
            print('synchronized', flush=True)
            compile_run_s = time.perf_counter() - started
            expected = reference(q, k, v, idx, bs, causal)
            torch.testing.assert_close(output, expected.to(output.dtype), atol=1e-2, rtol=1e-2)
            max_error = (output.float() - expected).abs().max().item()
            mod.run_kernel(q, k, v, idx, output, *case)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            mod.run_kernel(q, k, v, idx, output, *case)
            end.record()
            end.synchronize()
            print(json.dumps(dict(case=case, seed=seed, status='PASS', max_error=max_error,
                                  first_call_s=compile_run_s, exploratory_ms=start.elapsed_time(end))), flush=True)


if __name__ == '__main__':
    main()
