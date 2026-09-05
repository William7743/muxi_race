#!/usr/bin/env python3
"""Small GPU correctness/edge checks for v723; NOT submission or benchmark code.

Uses the existing remote_bench loader/invoker. No timing or performance claims.
CPU-generated inputs avoid device-RNG issues. Output equality does not prove
absence of speculative OOB loads: also inspect generated addresses separately.
"""
import argparse
from pathlib import Path

import torch

from remote_bench import invoke, load_submission


def make_tensors(sizes, weights_dtype, *, padded_override=None, hidden=128, intermediate=128):
    experts = 32
    assert len(sizes) == experts
    sizes_cpu = torch.tensor(sizes, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), sizes_cpu.cumsum(0).to(torch.int32)])
    padded_sizes = ((sizes_cpu + 127) // 128) * 128
    padded_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32), padded_sizes.cumsum(0).to(torch.int32)]
    )
    block_map = torch.repeat_interleave(
        torch.arange(experts, dtype=torch.int32), padded_sizes // 128
    )
    padded = int(padded_offsets[-1]) if padded_override is None else padded_override
    valid = sum(sizes)
    if padded_override is not None:
        assert valid == 0
        # Deliberately empty block map: the empty-route kernel must ignore it.
        block_map = torch.empty((0,), dtype=torch.int32)
        padded_offsets[1:] = padded

    generator = torch.Generator(device="cpu").manual_seed(723)
    def data(shape, scale=1.0):
        value = torch.randn(shape, generator=generator, dtype=torch.float16)
        value.mul_(scale)
        if valid == 0:
            value.fill_(float("nan"))
        return value.to("cuda")

    x = data((padded, hidden), 0.1)
    gate = data((experts, intermediate, hidden), 0.1)
    up = data((experts, intermediate, hidden), 0.1)
    down = data((experts, hidden, intermediate), 0.1)
    routed = torch.linspace(0.25, 0.75, steps=valid, dtype=torch.float32).to(weights_dtype).to("cuda")
    tensors = (
        x, gate, up, down, routed, sizes_cpu.to("cuda"), offsets.to("cuda"),
        padded_offsets.to("cuda"), block_map.to("cuda"),
    )
    return tensors, padded


def reject(*args, **kwargs):
    raise AssertionError("Unexpected Stage1/workspace/JIT path on empty output")


def check_pad_zero(candidate, dtype):
    tensors, padded = make_tensors([0] * 32, dtype, padded_override=0)
    out = torch.empty((padded, 128), device="cuda", dtype=torch.float16)
    names = ("_get_workspace", "_get_stage1", "_get_stage2")
    saved = {name: getattr(candidate, name) for name in names}
    try:
        for name in names:
            setattr(candidate, name, reject)
        invoke(candidate, tensors, out)
        torch.cuda.synchronize()
    finally:
        for name, value in saved.items():
            setattr(candidate, name, value)
    assert out.numel() == 0
    print(f"PASS pad=0 dtype={dtype}: no workspace/build/launch path", flush=True)


def check_empty_routes(candidate, dtype):
    # Both dimensions cross a tile boundary; all input payloads are poisoned.
    tensors, padded = make_tensors(
        [0] * 32, dtype, padded_override=129, hidden=129, intermediate=128
    )
    out = torch.full((padded, 129), float("nan"), device="cuda", dtype=torch.float16)
    saved_stage1 = candidate._get_stage1
    candidate._get_stage1 = reject
    try:
        invoke(candidate, tensors, out)
        torch.cuda.synchronize()
    finally:
        candidate._get_stage1 = saved_stage1
    assert torch.count_nonzero(out).item() == 0, "Empty-route output was not fully overwritten with zeros"
    assert torch.isfinite(out).all().item()
    print(f"PASS valid=0 pad=129 hidden=129 dtype={dtype}: empty block map, no Stage1, all-zero output", flush=True)


def check_positive(candidate, baseline, dtype, sizes, label):
    tensors, padded = make_tensors(sizes, dtype)
    hidden = tensors[0].shape[1]
    outputs = []
    for module in (baseline, candidate):
        workspace = module._get_workspace(tensors[0], 128)
        workspace.fill_(float("nan"))
        out = torch.full((padded, hidden), float("nan"), device="cuda", dtype=torch.float16)
        invoke(module, tensors, out)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all().item(), f"{label}: nonfinite output"
        outputs.append(out)
    baseline_out, candidate_out = outputs
    assert torch.equal(baseline_out.view(torch.int16), candidate_out.view(torch.int16)), (
        f"{label} dtype={dtype}: not byte-identical; max_abs="
        f"{(baseline_out.float() - candidate_out.float()).abs().max().item()}"
    )
    # Final expert owns just one real token; its 127 padding rows follow the
    # final allocated raw route weight and must all remain explicitly zero.
    last_size = sizes[-1]
    assert last_size == 1
    assert torch.count_nonzero(candidate_out[-127:]).item() == 0
    assert torch.count_nonzero(candidate_out[-128]).item() > 0, "Real last token unexpectedly all-zero"
    print(
        f"PASS {label} valid={sum(sizes)} pad={padded} dtype={dtype}: "
        "bitwise v720 match, final raw token followed by 127 zero padding rows",
        flush=True,
    )


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(here / "probe_v723_v720_e32_route_load_bounds.py"))
    parser.add_argument("--baseline", default=str(here / "probe_v720_v719_e16_stage2_bfrag_only.py"))
    args = parser.parse_args()
    assert torch.cuda.is_available(), "GPU runtime required"
    baseline = load_submission(args.baseline, "v723_edge_baseline", kernel_suffix="v723_edge_base")
    candidate = load_submission(args.candidate, "v723_edge_candidate", kernel_suffix="v723_edge_probe")
    with torch.no_grad():
        for dtype in (torch.float16, torch.float32):
            check_pad_zero(candidate, dtype)
            check_empty_routes(candidate, dtype)
            check_positive(candidate, baseline, dtype, [128] + [0] * 30 + [1], "full_then_short_tail")
            check_positive(candidate, baseline, dtype, [0] * 31 + [1], "one_token_last_expert")
    print("PASS all 8 small edge checks; no benchmark/timing performed.", flush=True)


if __name__ == "__main__":
    main()
