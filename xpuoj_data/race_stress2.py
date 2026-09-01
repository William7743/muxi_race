#!/usr/bin/env python3
"""Race stress v2: bigger iteration counts + randomized group splits."""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/moe_contest")
import remote_bench as rb

BLOCK_M = 128


def random_group_sizes(cfg, gen):
    E, valid = cfg["experts"], cfg["valid"]
    sizes = []
    remaining = valid
    for _ in range(E - 1):
        lo = 0
        hi = min(remaining, 400)
        v = int(torch.randint(lo, hi + 1, (1,), generator=gen).item())
        sizes.append(v)
        remaining -= v
    sizes.append(remaining)
    return torch.tensor(sizes, dtype=torch.int32)


def reference(tensors, cfg, padded_total, device):
    x, gate, up, down, routed, gs, go, gpo, gidx = tensors
    H = cfg["hidden"]
    E = cfg["experts"]
    out = torch.zeros((padded_total, H), dtype=torch.float32, device=device)
    sizes = gs.cpu().tolist()
    offs = go.cpu().tolist()
    padds = gpo.cpu().tolist()
    x32 = x.float()
    for e in range(E):
        n = sizes[e]
        if n == 0:
            continue
        seg = x32[padds[e] : padds[e] + n]
        g = seg @ gate[e].float().T
        u = seg @ up[e].float().T
        act = torch.nn.functional.silu(g) * u
        d = act @ down[e].float().T
        out[padds[e] : padds[e] + n] = d * routed[offs[e] : offs[e] + n].float().unsqueeze(1)
    return out


def check(out, ref, tag):
    diff = (out.float() - ref).abs()
    tol = 0.05 + 0.05 * ref.abs()
    bad = diff > tol
    n = int(bad.sum())
    if n:
        idx = bad.nonzero()[0].tolist()
        i, j = idx
        print(f"[FAIL] {tag}: {n} bad, first=({i},{j}) out={out[i,j].item():.6f} ref={ref[i,j].item():.6f}", flush=True)
    return n


def build_custom_inputs(cfg, gs_cpu, seed, device):
    torch.manual_seed(seed)
    go = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(gs_cpu, 0, dtype=torch.int32)])
    padded = torch.div(gs_cpu + BLOCK_M - 1, BLOCK_M, rounding_mode="floor") * BLOCK_M
    gpo = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(padded, 0, dtype=torch.int32)])
    padded_total = int(gpo[-1])
    gidx = torch.repeat_interleave(torch.arange(cfg["experts"], dtype=torch.int32), padded // BLOCK_M)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn((padded_total, cfg["hidden"]), generator=gen, dtype=torch.float16)
    gate = torch.randn((cfg["experts"], cfg["intermediate"], cfg["hidden"]), generator=gen, dtype=torch.float16) * 0.02
    up = torch.randn((cfg["experts"], cfg["intermediate"], cfg["hidden"]), generator=gen, dtype=torch.float16) * 0.02
    down = torch.randn((cfg["experts"], cfg["hidden"], cfg["intermediate"]), generator=gen, dtype=torch.float16) * 0.02
    routed = torch.rand((int(gs_cpu.sum()),), generator=gen, dtype=torch.float16)
    return padded_total, (
        x.to(device), gate.to(device), up.to(device), down.to(device),
        routed.to(device), gs_cpu.to(device), go.to(device), gpo.to(device), gidx.to(device),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--case", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--suffix", default="cand")
    args = ap.parse_args()

    device = "cuda"
    cfg, padded_total, tensors = rb.make_inputs(args.case, 20260901 + args.case)
    ref = reference(tensors, cfg, padded_total, device)
    mod = rb.load_submission(str(Path(args.candidate).resolve()), "moe_cand", args.suffix)
    out = torch.empty((padded_total, cfg["hidden"]), device=device, dtype=torch.float16)

    def run_once():
        rb.invoke(mod, tensors, out)
        torch.cuda.synchronize()

    run_once()
    print("== Phase A: iteration race ==", flush=True)
    fails = 0
    snap = torch.empty_like(out)
    for it in range(args.iters):
        run_once()
        if it % 7 == 0:
            snap.copy_(out)
            n = check(snap, ref, f"iter{it}")
            fails += 1 if n else 0
    print(f"Phase A: {fails} snapshots failed of {args.iters//7}", flush=True)

    print("== Phase B: random group-split sweep ==", flush=True)
    gen = torch.Generator(device="cpu")
    bfail = 0
    for s in range(args.seeds):
        gen.manual_seed(999000 + s * 17)
        gs2 = random_group_sizes(cfg, gen)
        pt2, tensors2 = build_custom_inputs(cfg, gs2, 555000 + s * 19 + args.case, device)
        ref2 = reference(tensors2, cfg, pt2, device)
        old_t, old_pt = tensors, padded_total
        tensors, padded_total = tensors2, pt2
        for r in range(3):
            run_once()
            n = check(out, ref2, f"seed{s}.r{r}")
            if n:
                bfail += 1
                break
        tensors, padded_total = old_t, old_pt
        del tensors2, ref2
    print(f"Phase B: {bfail}/{args.seeds} splits failed", flush=True)
    print(f"TOTAL {args.candidate} case{args.case}: B={bfail}/{args.seeds}", flush=True)


if __name__ == "__main__":
    main()
