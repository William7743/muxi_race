"""Diagnostic-only E64 Stage1, isolated Stage2, combined and host-empty checks.

Reference: v745 Stage1 plus its explicitly selected clamped M128 Stage2 builder.
Candidate: v748 (Stage1 identical to v746, Stage2 identical to v747).
Torch only creates data/compares results; this helper is not submission code.
"""

import argparse
import hashlib
from pathlib import Path

import torch
import tilelang.language as T

from remote_bench import load_submission
from remote_v743_stage2_edges import SIZES as SIZES32

SIZES = SIZES32 * 2


def metadata():
    offsets, padded_offsets, blocks, invalid_rows, row_counts = [0], [0], [], [], []
    for expert, size in enumerate(SIZES):
        padded_size = max(128, ((size + 127) // 128) * 128)
        start = padded_offsets[-1]
        offsets.append(offsets[-1] + size)
        padded_offsets.append(start + padded_size)
        blocks.extend([expert] * (padded_size // 128))
        invalid_rows.extend(range(start + size, start + padded_size))
        row_counts.extend(max(0, min(128, size - k)) for k in range(0, padded_size, 128))
    assert len(SIZES) == 64
    assert all(n in row_counts for n in (0, 1, 63, 64, 65, 127, 128))
    return offsets, padded_offsets, blocks, invalid_rows, row_counts


def describe(label, output, reference):
    finite = bool(torch.isfinite(output).all() and torch.isfinite(reference).all())
    diff = (output.float() - reference.float()).abs()
    max_abs = float(diff.max())
    nonzero = int(torch.count_nonzero(diff))
    bitwise = bool(torch.equal(output.view(torch.int16), reference.view(torch.int16)))
    bad = int(torch.count_nonzero(~torch.isfinite(diff) | (diff > 0.05 + 0.05 * reference.float().abs())))
    print(
        f"{label} finite={finite} max_abs_full={max_abs:.17g} nonzero_diff={nonzero} "
        f"bitwise_equal={bitwise} tolerance_bad={bad}/{output.numel()}", flush=True,
    )
    assert finite and bad == 0
    return bitwise


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(here / "probe_v748_v747_e64_stage1_runtime_m64.py"))
    parser.add_argument("--baseline", default=str(here / "probe_v745_v743_e32_stage1_runtime_m64.py"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    for label, filename in (("baseline", args.baseline), ("candidate", args.candidate)):
        path = Path(filename).resolve(strict=True)
        print(f"{label}={path.name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    offsets, padded_offsets, blocks, invalid_rows, row_counts = metadata()
    padded, valid = padded_offsets[-1], offsets[-1]
    hidden, intermediate, experts = 7168, 2048, 64
    invalid_set = set(invalid_rows)
    valid_rows = [i for i in range(padded) if i not in invalid_set]
    assert len(valid_rows) == valid
    print(f"metadata E={experts} H={hidden} I={intermediate} raw={valid} padded={padded} "
          f"blocks={len(blocks)} rows={row_counts}; local fixture, not official OJ input", flush=True)
    meta_gpu = [torch.tensor(values, dtype=torch.int32, device="cuda")
                for values in (SIZES, offsets, padded_offsets, blocks)]
    valid_gpu = torch.tensor(valid_rows, dtype=torch.int64, device="cuda")
    invalid_gpu = torch.tensor(invalid_rows, dtype=torch.int64, device="cuda")

    def random_weight(label, shape, seed):
        print(f"allocating {label} seed={seed}", flush=True)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        cpu = torch.randn(shape, dtype=torch.float16, generator=generator)
        cpu.mul_(0.1)
        return cpu.to("cuda")

    gate = random_weight("gate", (experts, intermediate, hidden), 74810)
    up_weight = random_weight("up", (experts, intermediate, hidden), 74811)
    down = random_weight("down", (experts, hidden, intermediate), 74812)
    base = load_submission(args.baseline, "v748_edge_base", kernel_suffix="v748_edge_base")
    probe = load_submission(args.candidate, "v748_edge_probe", kernel_suffix="v748_edge_probe")
    stage1_base = base._get_stage1(hidden, intermediate, experts, padded, len(blocks))
    stage1_probe = probe._get_stage1(hidden, intermediate, experts, padded, len(blocks))
    work_base = torch.empty((padded, intermediate), device="cuda", dtype=torch.float16)
    work_probe = torch.empty_like(work_base)
    out_base = torch.empty((padded, hidden), device="cuda", dtype=torch.float16)
    out_probe = torch.empty_like(out_base)
    out_isolated = torch.empty_like(out_base)
    print('Reference Stage2: explicit clamped M128 builder, not v745 E64 dispatcher', flush=True)
    all_bitwise = True
    with torch.no_grad():
        for repeat in range(args.rounds):
            seed = 74801 + repeat
            generator = torch.Generator(device="cpu").manual_seed(seed)
            cpu_x = torch.randn((padded, hidden), generator=generator, dtype=torch.float16)
            cpu_x.mul_(0.1)
            cpu_x[invalid_rows] = float("nan")
            x = cpu_x.to("cuda")
            routes_cpu = torch.rand((valid,), generator=generator, dtype=torch.float32)
            routes_cpu.mul_(0.5).add_(0.25)
            work_base.fill_(float("nan"))
            work_probe.fill_(float("nan"))
            stage1_base(x, gate, up_weight, meta_gpu[0], meta_gpu[2], meta_gpu[3], work_base)
            stage1_probe(x, gate, up_weight, meta_gpu[0], meta_gpu[2], meta_gpu[3], work_probe)
            torch.cuda.synchronize()
            all_bitwise &= describe(f"stage1 repeat={repeat + 1} seed={seed}",
                                    work_probe.index_select(0, valid_gpu), work_base.index_select(0, valid_gpu))
            untouched = bool(torch.isnan(work_base.index_select(0, invalid_gpu)).all()
                             and torch.isnan(work_probe.index_select(0, invalid_gpu)).all())
            print(f"stage1 repeat={repeat + 1} padding_untouched_nan={untouched}", flush=True)
            assert untouched
            for dtype, tl_dtype in ((torch.float32, T.float32), (torch.float16, T.float16)):
                routes = routes_cpu.to(dtype).to("cuda")
                stage2_base = base._moe_stage2_fast_bfrag_prefetch_route_bounds(
                    hidden, intermediate, experts, padded, valid, len(blocks), 128, 128, 64, 256, tl_dtype)
                stage2_probe = probe._get_stage2(hidden, intermediate, experts, padded, valid, len(blocks), tl_dtype)
                out_base.fill_(float("nan"))
                out_probe.fill_(float("nan"))
                out_isolated.fill_(float("nan"))
                stage2_base(work_base, down, routes, *meta_gpu, out_base)
                stage2_probe(work_base, down, routes, *meta_gpu, out_isolated)
                stage2_probe(work_probe, down, routes, *meta_gpu, out_probe)
                torch.cuda.synchronize()
                all_bitwise &= describe(f"isolated_stage2 repeat={repeat + 1} dtype={dtype}", out_isolated, out_base)
                all_bitwise &= describe(f"chain repeat={repeat + 1} dtype={dtype}", out_probe, out_base)
                zero_padding = not bool(torch.count_nonzero(out_probe.index_select(0, invalid_gpu)))
                zero_padding &= not bool(torch.count_nonzero(out_isolated.index_select(0, invalid_gpu)))
                print(f"chain repeat={repeat + 1} dtype={dtype} zero_padding={zero_padding}", flush=True)
                assert zero_padding
                assert int(torch.count_nonzero(out_probe[-128])) > 0
                del routes
            del cpu_x, x, routes_cpu

    # Diagnostic instrumentation only: empty inputs must not request Stage1.
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected getter/workspace request in an empty-input host path")

    original_stage1 = probe._get_stage1
    original_workspace = probe._get_workspace
    try:
        probe._get_stage1 = forbidden
        zero_sizes = torch.zeros_like(meta_gpu[0])
        zero_offsets = torch.zeros_like(meta_gpu[1])
        poisoned_x = torch.full((padded, hidden), float("nan"), device="cuda", dtype=torch.float16)
        for dtype in (torch.float32, torch.float16):
            empty_routes = torch.empty((0,), device="cuda", dtype=dtype)
            out_probe.fill_(float("nan"))
            probe.run_kernel(poisoned_x, gate, up_weight, down, empty_routes,
                             zero_sizes, zero_offsets, meta_gpu[2], meta_gpu[3], out_probe)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(out_probe).all())
            nonzero = int(torch.count_nonzero(out_probe))
            print(f"empty_routes dtype={dtype} finite={finite} nonzero={nonzero} stage1_getter_skipped=True", flush=True)
            assert finite and nonzero == 0
        probe._get_workspace = forbidden
        empty_x = torch.empty((0, hidden), device="cuda", dtype=torch.float16)
        empty_out = torch.empty_like(empty_x)
        empty_blocks = torch.empty((0,), device="cuda", dtype=torch.int32)
        for dtype in (torch.float32, torch.float16):
            empty_routes = torch.empty((0,), device="cuda", dtype=dtype)
            probe.run_kernel(empty_x, gate, up_weight, down, empty_routes,
                             zero_sizes, zero_offsets, zero_offsets, empty_blocks, empty_out)
            torch.cuda.synchronize()
            assert empty_out.numel() == 0
            print(f"empty_output dtype={dtype} return_pass=True workspace_getter_skipped=True", flush=True)
    finally:
        probe._get_stage1 = original_stage1
        probe._get_workspace = original_workspace

    print(f"PASS edge tolerance checks; all_tested_bitwise={all_bitwise}; baseline comparison, "
          "not independent mathematical/OJ validation or timing.", flush=True)


if __name__ == "__main__":
    main()
