#!/usr/bin/env python3
"""Judge-like C500 correctness/performance harness for Fused MoE candidates.

Each process imports exactly one TileLang submission and runs one public shape.
Use ``--save-output`` with a trusted OJ-Accepted source, then run a candidate in
a fresh process with ``--reference-output``.  This avoids JIT symbol/cache
collisions between submissions that use identical primfunc names.
"""

import argparse
import ast
import math
from pathlib import Path
import types

import torch
import tilelang.language as T


BLOCK_M = 128
CASES = {
    0: dict(experts=1, hidden=128, intermediate=128, valid=127, high_experts=0, low_size=127),
    4: dict(experts=1, hidden=2048, intermediate=8192, valid=3072, high_experts=1, low_size=0),
    5: dict(experts=1, hidden=2048, intermediate=8192, valid=127, high_experts=0, low_size=127),
    6: dict(experts=1, hidden=2048, intermediate=128, valid=3072, high_experts=1, low_size=0),
    1: dict(experts=16, hidden=2048, intermediate=8192, valid=2272, high_experts=8, low_size=124),
    2: dict(experts=32, hidden=7168, intermediate=2048, valid=4544, high_experts=22, low_size=112),
    3: dict(experts=64, hidden=7168, intermediate=2048, valid=9088, high_experts=23, low_size=127),
}


def load_submission(path: str, module_name: str, kernel_suffix: str = ""):
    source = Path(path).read_text()
    syntax_tree = ast.parse(source, filename=path)
    if kernel_suffix:
        # TileLang 0.1.10's process-global memory cache keys nested primfuncs
        # by name. Give every candidate unique, semantics-neutral names so a
        # batch run cannot accidentally reuse the first candidate's kernels.
        rename = {
            kernel_name: f"{kernel_name}_{kernel_suffix}"
            for kernel_name in ("stage1", "stage2")
        }

        class RenameKernels(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.name = rename.get(node.name, node.name)
                return self.generic_visit(node)

            def visit_Name(self, node):
                node.id = rename.get(node.id, node.id)
                return node

        syntax_tree = ast.fix_missing_locations(RenameKernels().visit(syntax_tree))
    module = types.ModuleType(module_name)
    module.__file__ = path
    exec(compile(syntax_tree, path, "exec"), module.__dict__)
    return module


def make_group_sizes(cfg):
    high = cfg["high_experts"]
    low = cfg["experts"] - high
    sizes = [cfg["low_size"]] * low
    remaining = cfg["valid"] - sum(sizes)
    if high:
        q, r = divmod(remaining, high)
        sizes.extend([q + (i < r) for i in range(high)])
    else:
        assert remaining == 0
    # Interleave large and small experts so swizzle/L2 behavior is not biased
    # by putting every two-block expert into one contiguous grid region.
    sizes = sizes[::2] + sizes[1::2]
    assert len(sizes) == cfg["experts"] and sum(sizes) == cfg["valid"]
    assert sum(x > BLOCK_M for x in sizes) == high
    return torch.tensor(sizes, dtype=torch.int32)


def make_inputs(case_id: int, seed: int):
    cfg = CASES[case_id]
    torch.manual_seed(seed)

    group_sizes_cpu = make_group_sizes(cfg)
    group_offsets_cpu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(group_sizes_cpu, 0, dtype=torch.int32),
        ]
    )
    padded_sizes = torch.div(
        group_sizes_cpu + BLOCK_M - 1, BLOCK_M, rounding_mode="floor"
    ) * BLOCK_M
    group_padded_offsets_cpu = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(padded_sizes, 0, dtype=torch.int32),
        ]
    )
    padded_total = int(group_padded_offsets_cpu[-1])
    group_idx_cpu = torch.repeat_interleave(
        torch.arange(cfg["experts"], dtype=torch.int32), padded_sizes // BLOCK_M
    )

    # C500's device RNG has intermittently segfaulted in this exact runtime.
    # Generate one tensor at a time on CPU and transfer it to the device.
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    def randn_cuda(label, shape, scale=1.0):
        print(f"allocating {label} on CPU: {shape}", flush=True)
        value = torch.randn(shape, generator=cpu_gen, dtype=torch.float16)
        if scale != 1.0:
            value.mul_(scale)
        result = value.to("cuda")
        print(f"allocated {label} on GPU", flush=True)
        return result

    device = "cuda"
    x = randn_cuda("x", (padded_total, cfg["hidden"]))
    gate = randn_cuda(
        "gate",
        (cfg["experts"], cfg["intermediate"], cfg["hidden"]),
        1.0 / math.sqrt(cfg["intermediate"]),
    )
    up = randn_cuda(
        "up",
        (cfg["experts"], cfg["intermediate"], cfg["hidden"]),
        1.0 / math.sqrt(cfg["intermediate"]),
    )
    down = randn_cuda(
        "down",
        (cfg["experts"], cfg["hidden"], cfg["intermediate"]),
        1.0 / math.sqrt(cfg["hidden"]),
    )
    print("allocating routed weights", flush=True)
    routed = torch.rand((cfg["valid"],), generator=cpu_gen, dtype=torch.float32).to(device)
    print("allocated routed weights", flush=True)

    print("copying metadata", flush=True)
    group_sizes = group_sizes_cpu.to(device)
    group_offsets = group_offsets_cpu.to(device)
    group_padded_offsets = group_padded_offsets_cpu.to(device)
    group_idx = group_idx_cpu.to(device)
    torch.cuda.synchronize()
    print("inputs ready", flush=True)
    return cfg, padded_total, (
        x,
        gate,
        up,
        down,
        routed,
        group_sizes,
        group_offsets,
        group_padded_offsets,
        group_idx,
    )


def invoke(module, tensors, out):
    module.run_kernel(*tensors, out)


def measure(module, tensors, out, warmup: int, iters: int):
    for _ in range(warmup):
        invoke(module, tensors, out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        invoke(module, tensors, out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, choices=sorted(CASES), required=True)
    parser.add_argument("--candidate", nargs="+", default=["submission.py"])
    parser.add_argument("--save-output")
    parser.add_argument("--reference-output")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--phase", choices=("full", "stage1", "stage2"), default="full")
    args = parser.parse_args()

    candidate_paths = [str(Path(path).resolve()) for path in args.candidate]
    if args.phase != "full" and len(candidate_paths) != 1:
        parser.error("stage1/stage2 phase accepts exactly one candidate")
    cfg, padded_total, tensors = make_inputs(args.case, args.seed + args.case)
    candidate_out = torch.empty(
        (padded_total, cfg["hidden"]), device="cuda", dtype=torch.float16
    )

    if args.phase == "stage1":
        candidate = load_submission(candidate_paths[0], "moe_candidate_phase")
        x, gate, up, _down, _routed, group_sizes, _go, gpo, gidx = tensors
        stage1 = candidate._get_stage1(
            cfg["hidden"], cfg["intermediate"], cfg["experts"], padded_total, int(gidx.numel())
        )
        workspace = candidate._get_workspace(x, cfg["intermediate"])
        print("launching stage1", flush=True)
        stage1(x, gate, up, group_sizes, gpo, gidx, workspace)
        torch.cuda.synchronize()
        print("stage1 OK", flush=True)
        return

    if args.phase == "stage2":
        candidate = load_submission(candidate_paths[0], "moe_candidate_phase")
        x, _gate, _up, down, routed, group_sizes, go, gpo, gidx = tensors
        workspace = torch.randn(
            (padded_total, cfg["intermediate"]), dtype=torch.float16
        ).to("cuda")
        stage2 = candidate._get_stage2(
            cfg["hidden"],
            cfg["intermediate"],
            cfg["experts"],
            padded_total,
            cfg["valid"],
            int(gidx.numel()),
            T.float32,
        )
        print("launching stage2", flush=True)
        stage2(workspace, down, routed, group_sizes, go, gpo, gidx, candidate_out)
        torch.cuda.synchronize()
        print("stage2 OK", flush=True)
        return

    if args.save_output and len(candidate_paths) != 1:
        parser.error("--save-output accepts exactly one candidate")

    reference_path = Path(args.reference_output).resolve() if args.reference_output else None
    reference_out = None
    if reference_path:
        reference_out = torch.load(reference_path, map_location="cuda", weights_only=True)
        if tuple(reference_out.shape) != tuple(candidate_out.shape):
            raise RuntimeError(
                f"reference shape {tuple(reference_out.shape)} != {tuple(candidate_out.shape)}"
            )

    for candidate_index, candidate_path in enumerate(candidate_paths):
        candidate = load_submission(
            candidate_path,
            f"moe_candidate_{candidate_index}",
            kernel_suffix=f"candidate_{candidate_index}",
        )
        candidate_ms = measure(candidate, tensors, candidate_out, args.warmup, args.iters)
        print(
            f"candidate={candidate_path} case={args.case} padded={padded_total} "
            f"blocks={padded_total // BLOCK_M} candidate_ms={candidate_ms:.6f}"
        )

        if args.save_output:
            output_path = Path(args.save_output).resolve()
            torch.save(candidate_out.cpu(), output_path)
            print(f"saved_output={output_path}")

        if reference_out is None:
            continue

        diff = (candidate_out.float() - reference_out.float()).abs()
        tol = 0.05 + 0.05 * reference_out.float().abs()
        bad = diff > tol
        max_abs = float(diff.max())
        bad_count = int(bad.sum())
        total = bad.numel()
        print(
            f"reference={reference_path} max_abs={max_abs:.6f} "
            f"bad={bad_count}/{total}"
        )
        if bad_count:
            first = int(torch.nonzero(bad, as_tuple=False)[0, 0])
            first_col = int(torch.nonzero(bad, as_tuple=False)[0, 1])
            print(
                f"first_bad=({first},{first_col}) got={float(candidate_out[first, first_col])} "
                f"ref={float(reference_out[first, first_col])} "
                f"diff={float(diff[first, first_col])}"
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
