"""Diagnostic-only Stage2 branch/route-boundary checks; never submission code.

Use actual E32/H7168/I2048 dimensions. Compare current-input outputs with v723,
including M64/M128 thresholds, explicit zero-row CTAs and the final raw token.
Two independent CPU random input sets per route dtype; no benchmark claims.
Torch is used only in this diagnostic for data generation and comparisons.
"""

import argparse
from pathlib import Path

import torch
import tilelang.language as T

from remote_bench import load_submission


SIZES = [
    0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65,
    79, 80, 81, 95, 96, 97, 111, 112, 113, 127, 128, 129, 191, 192, 255, 1,
]


def metadata():
    offsets = [0]
    padded_offsets = [0]
    blocks = []
    invalid_rows = []
    row_counts = []
    for expert, size in enumerate(SIZES):
        # The first expert deliberately owns an allocated empty M128 block.
        padded_size = max(128, ((size + 127) // 128) * 128)
        start = padded_offsets[-1]
        offsets.append(offsets[-1] + size)
        padded_offsets.append(start + padded_size)
        blocks.extend([expert] * (padded_size // 128))
        invalid_rows.extend(range(start + size, start + padded_size))
        row_counts.extend(max(0, min(128, size - k)) for k in range(0, padded_size, 128))
    assert len(SIZES) == 32
    assert all(n in row_counts for n in (0, 1, 63, 64, 65, 127, 128))
    return offsets, padded_offsets, blocks, invalid_rows, row_counts


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=str(here / "probe_v743_v723_e32_stage2_runtime_m64.py"))
    parser.add_argument("--baseline", default=str(here / "probe_v723_v720_e32_route_load_bounds.py"))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    offsets, padded_offsets, blocks, invalid_rows, row_counts = metadata()
    padded, valid = padded_offsets[-1], offsets[-1]
    hidden, intermediate, experts = 7168, 2048, 32
    print(
        f"metadata E={experts} H={hidden} I={intermediate} raw={valid} padded={padded} "
        f"blocks={len(blocks)} rows={row_counts}; synthetic, not official OJ inputs",
        flush=True,
    )
    baseline = load_submission(args.baseline, "v743_edge_base", kernel_suffix="v743_edge_base")
    candidate = load_submission(args.candidate, "v743_edge_probe", kernel_suffix="v743_edge_probe")
    meta_gpu = [
        torch.tensor(values, dtype=torch.int32, device="cuda")
        for values in (SIZES, offsets, padded_offsets, blocks)
    ]
    invalid_gpu = torch.tensor(invalid_rows, dtype=torch.int64, device="cuda")
    down_generator = torch.Generator(device="cpu").manual_seed(74300)
    print("allocating random down weights", flush=True)
    down_cpu = torch.randn((experts, hidden, intermediate), generator=down_generator, dtype=torch.float16)
    down_cpu.mul_(0.1)
    down = down_cpu.to("cuda")
    del down_cpu
    reference = torch.empty((padded, hidden), device="cuda", dtype=torch.float16)
    output = torch.empty_like(reference)
    with torch.no_grad():
        for dtype, tl_dtype in ((torch.float32, T.float32), (torch.float16, T.float16)):
            base_kernel = baseline._get_stage2(
                hidden, intermediate, experts, padded, valid, len(blocks), tl_dtype,
            )
            new_kernel = candidate._get_stage2(
                hidden, intermediate, experts, padded, valid, len(blocks), tl_dtype,
            )
            for repeat in range(args.rounds):
                seed = 74301 + repeat
                generator = torch.Generator(device="cpu").manual_seed(seed)
                up_cpu = torch.randn((padded, intermediate), generator=generator, dtype=torch.float16)
                up_cpu.mul_(0.1)
                up_cpu[invalid_rows] = float("nan")
                up = up_cpu.to("cuda")
                routes = torch.rand((valid,), generator=generator, dtype=torch.float32)
                routes.mul_(0.5).add_(0.25)
                routes = routes.to(dtype).to("cuda")
                reference.fill_(float("nan"))
                output.fill_(float("nan"))
                base_kernel(up, down, routes, *meta_gpu, reference)
                new_kernel(up, down, routes, *meta_gpu, output)
                torch.cuda.synchronize()
                finite = bool(torch.isfinite(reference).all() and torch.isfinite(output).all())
                diff = (output.float() - reference.float()).abs()
                max_abs = float(diff.max())
                nonzero = int(torch.count_nonzero(diff))
                bitwise = bool(torch.equal(output.view(torch.int16), reference.view(torch.int16)))
                padding_nonzero = int(torch.count_nonzero(output.index_select(0, invalid_gpu)))
                print(
                    f"check dtype={dtype} repeat={repeat + 1} seed={seed} finite={finite} "
                    f"max_abs_full={max_abs:.17g} nonzero_diff={nonzero} "
                    f"bitwise_equal={bitwise} padding_nonzero={padding_nonzero}",
                    flush=True,
                )
                assert finite and bitwise and nonzero == 0 and padding_nonzero == 0
                assert int(torch.count_nonzero(output[-128])) > 0, "Final valid token unexpectedly all zero"
                del up, up_cpu, routes, diff
    print("PASS Stage2-only edge checks; baseline comparison, not independent mathematical/OJ reference.", flush=True)


if __name__ == "__main__":
    main()
