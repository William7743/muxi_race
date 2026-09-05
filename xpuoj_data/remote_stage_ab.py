#!/usr/bin/env python3
"""Interleaved remote A/B benchmark for split Fused-MoE stages.

The harness reuses :mod:`remote_bench` for deterministic inputs and isolated
submission loading.  Every candidate is compiled once, checked end-to-end
against the saved v432 reference, and then timed in alternating forward/reverse
order to reduce bias from clock drift.

Example
-------
python remote_stage_ab.py \
    --case 2 \
    --stage s2 \
    --candidates probe_v527.py probe_v545.py
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import types
from typing import Callable

import torch
import tilelang.language as T

import remote_bench as rb


DEFAULT_SEED = 20260901

# Alternate local routing fixture. Its historical "oj-real" label was not
# backed by a recoverable official testcase generator; do not treat it as a
# verified OJ distribution. Keep the old CLI name only as a compatibility alias.
ALTERNATING_GROUP_SIZES = {
    1: tuple([64, 220] * 8),
    2: tuple([64, 220] * 16),
    3: tuple([64, 220] * 32),
}


@dataclass(frozen=True)
class CompiledCandidate:
    """One loaded candidate and its already-built stage callables."""

    index: int
    path: Path
    module: types.ModuleType
    stage1: Callable[..., object] | tuple[Callable[..., object], ...]
    stage2: Callable[..., object] | tuple[Callable[..., object], ...]

    @property
    def label(self) -> str:
        return f"c{self.index}:{self.path.name}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness-first, interleaved Stage1/Stage2/full A/B benchmark."
    )
    parser.add_argument("--case", type=int, choices=sorted(rb.CASES), required=True)
    parser.add_argument("--stage", choices=("s1", "s2", "full", "entry", "all"), required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument(
        "--routing",
        choices=("synthetic", "alternating64-220", "oj-real"),
        default="synthetic",
        help="local routing fixture; oj-real is a legacy alias, not verified OJ data",
    )
    parser.add_argument(
        "--input-mode",
        choices=("random", "constant"),
        default="random",
        help="constant fills tensors directly on GPU for rapid scheduling probes",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iters", type=positive_int, default=3)
    parser.add_argument("--rounds", type=positive_int, default=4)
    parser.add_argument(
        "--correctness-repeats",
        type=positive_int,
        default=1,
        help="repeat the complete correctness path to expose nondeterministic kernels",
    )
    parser.add_argument(
        "--verify-run-kernel",
        action="store_true",
        help="also check the real run_kernel entrypoint outside benchmark timing",
    )
    return parser.parse_args()


def make_constant_inputs(case_id: int, routing: str) -> tuple:
    """Allocate inexpensive nonzero tensors directly on GPU for timing sweeps."""
    cfg = rb.CASES[case_id]
    if routing in ("alternating64-220", "oj-real"):
        sizes = ALTERNATING_GROUP_SIZES.get(case_id)
        if sizes is None:
            raise ValueError("alternating routing is defined only for cases 1, 2 and 3")
        group_sizes_cpu = torch.tensor(sizes, dtype=torch.int32)
    else:
        group_sizes_cpu = rb.make_group_sizes(cfg)

    group_offsets_cpu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(group_sizes_cpu, 0, dtype=torch.int32),
        )
    )
    padded_sizes = torch.div(
        group_sizes_cpu + rb.BLOCK_M - 1,
        rb.BLOCK_M,
        rounding_mode="floor",
    ) * rb.BLOCK_M
    group_padded_offsets_cpu = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(padded_sizes, 0, dtype=torch.int32),
        )
    )
    padded_total = int(group_padded_offsets_cpu[-1])
    group_idx_cpu = torch.repeat_interleave(
        torch.arange(cfg["experts"], dtype=torch.int32),
        padded_sizes // rb.BLOCK_M,
    )

    def full_cuda(label: str, shape: tuple[int, ...], value: float, dtype=torch.float16):
        print(f"allocating constant {label} on GPU: {shape}", flush=True)
        result = torch.full(shape, value, dtype=dtype, device="cuda")
        print(f"allocated constant {label}", flush=True)
        return result

    x = full_cuda("x", (padded_total, cfg["hidden"]), 0.01)
    gate = full_cuda("gate", (cfg["experts"], cfg["intermediate"], cfg["hidden"]), 0.001)
    up = full_cuda("up", (cfg["experts"], cfg["intermediate"], cfg["hidden"]), 0.0015)
    down = full_cuda(
        "down",
        (cfg["experts"], cfg["hidden"], cfg["intermediate"]),
        0.0005,
    )
    routed = full_cuda("routed weights", (cfg["valid"],), 0.5, dtype=torch.float32)
    group_sizes = group_sizes_cpu.to("cuda")
    group_offsets = group_offsets_cpu.to("cuda")
    group_padded_offsets = group_padded_offsets_cpu.to("cuda")
    group_idx = group_idx_cpu.to("cuda")
    torch.cuda.synchronize()
    print("constant inputs ready", flush=True)
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


def make_inputs(case_id: int, routing: str, input_mode: str) -> tuple:
    """Build inputs with either the historical skew or public OJ telemetry."""
    if input_mode == "constant":
        return make_constant_inputs(case_id, routing)
    if routing == "synthetic":
        return rb.make_inputs(case_id, DEFAULT_SEED + case_id)

    sizes = ALTERNATING_GROUP_SIZES.get(case_id)
    if sizes is None:
        raise ValueError("alternating routing is defined only for cases 1, 2 and 3")
    cfg = rb.CASES[case_id]
    if len(sizes) != cfg["experts"] or sum(sizes) != cfg["valid"]:
        raise RuntimeError(f"invalid alternating routing for case {case_id}: {sizes}")

    original_make_group_sizes = rb.make_group_sizes
    rb.make_group_sizes = lambda _cfg: torch.tensor(sizes, dtype=torch.int32)
    try:
        return rb.make_inputs(case_id, DEFAULT_SEED + case_id)
    finally:
        rb.make_group_sizes = original_make_group_sizes


def resolve_candidates(raw_paths: list[str]) -> list[Path]:
    paths = [Path(raw_path).expanduser().resolve() for raw_path in raw_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("candidate file(s) not found: " + ", ".join(missing))
    return paths


def get_weights_dtype(routed: torch.Tensor):
    return T.float32 if routed.dtype == torch.float32 else T.float16


def resolve_stage2_builder(module: types.ModuleType, num_experts: int) -> str:
    """Prefer a unique three-scope builder over the inherited two-scope builder."""
    pattern = re.compile(rf"_get_stage2_e{num_experts}_middle_split_v\d+")
    middle_names = sorted(name for name in vars(module) if pattern.fullmatch(name))
    if len(middle_names) > 1:
        raise RuntimeError(
            f"ambiguous Stage2 middle builders in {module.__file__}: {middle_names}"
        )
    if middle_names:
        return middle_names[0]
    split_name = f"_get_stage2_e{num_experts}_split"
    return split_name if hasattr(module, split_name) else "_get_stage2"


def compile_candidates(
    paths: list[Path],
    cfg: dict,
    padded_total: int,
    tensors: tuple,
) -> list[CompiledCandidate]:
    """Load each source and build each selected-shape stage exactly once."""
    *_, routed, _group_sizes, _group_offsets, _gpo, group_idx = tensors
    num_blocks_m = int(group_idx.numel())
    weights_dtype = get_weights_dtype(routed)
    compiled = []

    for index, path in enumerate(paths):
        print(f"loading candidate={index} path={path}", flush=True)
        module = rb.load_submission(
            str(path),
            f"moe_stage_ab_candidate_{index}",
            kernel_suffix=f"stage_ab_case_candidate_{index}",
        )
        for required_name in ("_get_stage1", "_get_stage2"):
            if not hasattr(module, required_name):
                raise AttributeError(f"{path} does not define {required_name}")

        print(f"compiling candidate={index} stage=s1", flush=True)
        stage1_args = (
            cfg["hidden"],
            cfg["intermediate"],
            cfg["experts"],
            padded_total,
            num_blocks_m,
        )
        stage1_split_name = f"_get_stage1_e{cfg['experts']}_split"
        if hasattr(module, stage1_split_name):
            stage1 = tuple(getattr(module, stage1_split_name)(*stage1_args))
        else:
            stage1 = module._get_stage1(*stage1_args)

        print(f"compiling candidate={index} stage=s2", flush=True)
        stage2_args = (
            cfg["hidden"],
            cfg["intermediate"],
            cfg["experts"],
            padded_total,
            int(routed.numel()),
            num_blocks_m,
            weights_dtype,
        )
        stage2_builder_name = resolve_stage2_builder(module, cfg["experts"])
        stage2 = getattr(module, stage2_builder_name)(*stage2_args)
        if stage2_builder_name != "_get_stage2":
            if not isinstance(stage2, tuple) or not stage2 or not all(
                callable(stage) for stage in stage2
            ):
                raise TypeError(
                    f"{path}: {stage2_builder_name} must return a nonempty callable tuple"
                )
        print(
            f"stage2_builder candidate={index} name={stage2_builder_name} "
            f"launches={len(stage2) if isinstance(stage2, tuple) else 1}",
            flush=True,
        )
        torch.cuda.synchronize()
        compiled.append(
            CompiledCandidate(
                index=index,
                path=path,
                module=module,
                stage1=stage1,
                stage2=stage2,
            )
        )
        print(f"compiled candidate={index}", flush=True)

    return compiled


def launch_stage1(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
) -> None:
    x, gate, up, _down, _routed, group_sizes, _go, gpo, group_idx = tensors
    stages = candidate.stage1 if isinstance(candidate.stage1, tuple) else (candidate.stage1,)
    for stage in stages:
        stage(x, gate, up, group_sizes, gpo, group_idx, workspace)


def launch_stage2(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _x, _gate, _up, down, routed, group_sizes, group_offsets, gpo, group_idx = tensors
    stages = candidate.stage2 if isinstance(candidate.stage2, tuple) else (candidate.stage2,)
    for stage in stages:
        stage(
            workspace,
            down,
            routed,
            group_sizes,
            group_offsets,
            gpo,
            group_idx,
            out,
        )


def launch_full(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> None:
    launch_stage1(candidate, tensors, workspace)
    launch_stage2(candidate, tensors, workspace, out)


def load_reference(case_id: int, expected_shape: tuple[int, ...]) -> tuple[Path, torch.Tensor]:
    reference_path = Path(f"/root/ref_v432_case{case_id}.pt")
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference output not found: {reference_path}")
    reference = torch.load(reference_path, map_location="cuda", weights_only=True)
    if not isinstance(reference, torch.Tensor):
        raise TypeError(f"reference is not a tensor: {type(reference)!r}")
    if tuple(reference.shape) != expected_shape:
        raise RuntimeError(
            f"reference shape {tuple(reference.shape)} != expected {expected_shape}"
        )
    return reference_path, reference


def compare_output(
    candidate: CompiledCandidate,
    out: torch.Tensor,
    reference_path: Path,
    reference: torch.Tensor,
    path_name: str,
) -> None:
    """Check shape, finiteness and remote_bench.py's numerical tolerance."""
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != tuple(reference.shape):
        raise RuntimeError(
            f"correctness {path_name} shape failed for {candidate.path}: "
            f"got={getattr(out, 'shape', None)} expected={tuple(reference.shape)}"
        )
    reference_f32 = reference.float()
    diff = (out.float() - reference_f32).abs()
    bad = (~torch.isfinite(out)) | (~torch.isfinite(reference_f32))
    bad |= diff > (0.05 + 0.05 * reference_f32.abs())
    max_abs = float(diff.max())
    bad_count = int(bad.sum())
    total = bad.numel()
    print(
        f"correctness candidate={candidate.label} path={path_name} "
        f"reference={reference_path} "
        f"max_abs={max_abs:.6f} bad={bad_count}/{total}",
        flush=True,
    )
    if not bad_count:
        return

    first_bad = torch.nonzero(bad, as_tuple=False)[0]
    row = int(first_bad[0])
    col = int(first_bad[1])
    raise RuntimeError(
        f"correctness {path_name} failed for {candidate.path}: first_bad=({row},{col}) "
        f"got={float(out[row, col])} ref={float(reference[row, col])} "
        f"diff={float(diff[row, col])}"
    )


def poison_buffers(workspace: torch.Tensor, out: torch.Tensor) -> None:
    """Expose skipped writes without adding any work to the timed launches."""
    workspace.fill_(float("nan"))
    out.fill_(float("nan"))
    # Stage1 may legitimately omit padded rows. Stage2 GEMM does not mix M rows,
    # and its epilogue explicitly writes zero for every padded output row; these
    # workspace NaNs must therefore never reach any output element.


def check_correctness(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
    out: torch.Tensor,
    reference_path: Path,
    reference: torch.Tensor,
    verify_run_kernel: bool = False,
) -> None:
    """Validate split launches, optionally followed by the submission entrypoint."""
    torch.cuda.synchronize()
    poison_buffers(workspace, out)
    launch_full(candidate, tensors, workspace, out)
    torch.cuda.synchronize()
    compare_output(candidate, out, reference_path, reference, "launch_full")

    if verify_run_kernel:
        # The submission contract takes nine input tensors and an in-place out
        # argument; it does not return a mapping/tuple of outputs. Exercise that
        # exact path so hand-selected split builders cannot hide dispatch bugs.
        entry_workspace = candidate.module._get_workspace(
            tensors[0], int(tensors[1].shape[1])
        )
        poison_buffers(entry_workspace, out)
        candidate.module.run_kernel(*tensors, out)
        torch.cuda.synchronize()
        compare_output(candidate, out, reference_path, reference, "run_kernel")


def measure_events(launch: Callable[[], None], iters: int) -> float:
    """Measure one candidate with an idle device before and after the event pair."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iters


def print_summary(
    candidates: list[CompiledCandidate],
    samples: dict[int, list[float]],
) -> None:
    baseline_median = statistics.median(samples[candidates[0].index])
    print("summary_begin", flush=True)
    for candidate in candidates:
        values = samples[candidate.index]
        mean_ms = statistics.mean(values)
        median_ms = statistics.median(values)
        stdev_ms = statistics.pstdev(values) if len(values) > 1 else 0.0
        speedup = (baseline_median / median_ms - 1.0) * 100.0
        sample_text = ",".join(f"{value:.6f}" for value in values)
        print(
            f"summary candidate={candidate.label} median_ms={median_ms:.6f} "
            f"mean_ms={mean_ms:.6f} stdev_ms={stdev_ms:.6f} "
            f"min_ms={min(values):.6f} max_ms={max(values):.6f} "
            f"speedup_vs_c0={speedup:+.3f}% samples_ms=[{sample_text}]",
            flush=True,
        )
    print("summary_end", flush=True)


def main() -> None:
    args = parse_args()
    if args.routing == "oj-real":
        print(
            "WARNING: oj-real is a legacy alias for a local alternating64-220 "
            "fixture, not a verified OJ testcase distribution.",
            flush=True,
        )
    candidate_paths = resolve_candidates(args.candidates)

    cfg, padded_total, tensors = make_inputs(args.case, args.routing, args.input_mode)
    expected_shape = (padded_total, cfg["hidden"])

    workspace = torch.empty(
        (padded_total, cfg["intermediate"]),
        device="cuda",
        dtype=torch.float16,
    )
    out = torch.empty(expected_shape, device="cuda", dtype=torch.float16)

    candidates = compile_candidates(candidate_paths, cfg, padded_total, tensors)

    if args.routing == "synthetic" and args.input_mode == "random":
        reference_path, reference = load_reference(args.case, expected_shape)
    else:
        # Candidate zero is the already-validated baseline (normally v552).
        # Materialize its result once so every experimental candidate is
        # checked on the exact same 64/220 metadata before timing.
        reference_path = Path(f"candidate_zero_{candidates[0].path.name}")
        reference_workspace = torch.empty_like(workspace)
        reference = torch.empty_like(out)
        poison_buffers(reference_workspace, reference)
        launch_full(candidates[0], tensors, reference_workspace, reference)
        torch.cuda.synchronize()
        reference = reference.clone()
        print(
            f"candidate_reference={candidates[0].label} routing={args.routing} "
            f"padded_total={padded_total} num_blocks_m={int(tensors[-1].numel())}",
            flush=True,
        )

    # Correctness always covers the complete Stage1 -> Stage2 path before any
    # performance numbers are produced.
    for correctness_round in range(args.correctness_repeats):
        for candidate in candidates:
            check_correctness(
                candidate,
                tensors,
                workspace,
                out,
                reference_path,
                reference,
                verify_run_kernel=args.verify_run_kernel,
            )
        print(
            f"correctness_round={correctness_round + 1}/{args.correctness_repeats}",
            flush=True,
        )
    print("all_candidates_correct", flush=True)

    fixed_stage2_workspace = None
    if args.stage in ("s2", "all"):
        # This buffer is written exactly once by candidate zero's Stage1 and is
        # thereafter passed unchanged to every candidate's Stage2.
        fixed_stage2_workspace = torch.empty_like(workspace)
        fixed_stage2_workspace.fill_(float("nan"))
        launch_stage1(candidates[0], tensors, fixed_stage2_workspace)
        torch.cuda.synchronize()
        print(
            f"fixed_stage2_workspace_source={candidates[0].label}",
            flush=True,
        )

    def selected_launch(stage_name: str, candidate: CompiledCandidate) -> None:
        if stage_name == "s1":
            launch_stage1(candidate, tensors, workspace)
        elif stage_name == "s2":
            if fixed_stage2_workspace is None:
                raise RuntimeError("fixed Stage2 workspace was not initialized")
            launch_stage2(candidate, tensors, fixed_stage2_workspace, out)
        elif stage_name == "entry":
            candidate.module.run_kernel(*tensors, out)
        else:
            launch_full(candidate, tensors, workspace, out)

    stage_names = ("s1", "s2", "full") if args.stage == "all" else (args.stage,)
    for stage_name in stage_names:
        print(f"stage_begin stage={stage_name}", flush=True)
        for candidate in candidates:
            for _ in range(args.warmup):
                selected_launch(stage_name, candidate)
            torch.cuda.synchronize()
            print(
                f"warmed stage={stage_name} candidate={candidate.label} "
                f"launches={args.warmup}",
                flush=True,
            )

        samples = {candidate.index: [] for candidate in candidates}
        for round_index in range(args.rounds):
            if round_index % 2 == 0:
                ordered = candidates
                order_name = "forward"
            else:
                ordered = list(reversed(candidates))
                order_name = "reverse"
            print(
                f"round={round_index + 1}/{args.rounds} stage={stage_name} "
                f"order={order_name}",
                flush=True,
            )
            for candidate in ordered:
                elapsed_ms = measure_events(
                    lambda current=candidate, current_stage=stage_name: selected_launch(
                        current_stage, current
                    ),
                    args.iters,
                )
                samples[candidate.index].append(elapsed_ms)
                print(
                    f"timing round={round_index + 1} order={order_name} "
                    f"stage={stage_name} candidate={candidate.label} "
                    f"iters={args.iters} ms={elapsed_ms:.6f}",
                    flush=True,
                )

        print(f"stage_summary stage={stage_name}", flush=True)
        print_summary(candidates, samples)
        print(f"stage_end stage={stage_name}", flush=True)


if __name__ == "__main__":
    main()
