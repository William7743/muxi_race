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
import statistics
import types
from typing import Callable

import torch
import tilelang.language as T

import remote_bench as rb


DEFAULT_SEED = 20260901


@dataclass(frozen=True)
class CompiledCandidate:
    """One loaded candidate and its already-built stage callables."""

    index: int
    path: Path
    module: types.ModuleType
    stage1: Callable[..., object]
    stage2: Callable[..., object]

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
    parser.add_argument("--stage", choices=("s1", "s2", "full"), required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--iters", type=positive_int, default=3)
    parser.add_argument("--rounds", type=positive_int, default=4)
    return parser.parse_args()


def resolve_candidates(raw_paths: list[str]) -> list[Path]:
    paths = [Path(raw_path).expanduser().resolve() for raw_path in raw_paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("candidate file(s) not found: " + ", ".join(missing))
    return paths


def get_weights_dtype(routed: torch.Tensor):
    return T.float32 if routed.dtype == torch.float32 else T.float16


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
        stage1 = module._get_stage1(
            cfg["hidden"],
            cfg["intermediate"],
            cfg["experts"],
            padded_total,
            num_blocks_m,
        )
        print(f"compiling candidate={index} stage=s2", flush=True)
        stage2 = module._get_stage2(
            cfg["hidden"],
            cfg["intermediate"],
            cfg["experts"],
            padded_total,
            int(routed.numel()),
            num_blocks_m,
            weights_dtype,
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
    candidate.stage1(x, gate, up, group_sizes, gpo, group_idx, workspace)


def launch_stage2(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _x, _gate, _up, down, routed, group_sizes, group_offsets, gpo, group_idx = tensors
    candidate.stage2(
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


def check_correctness(
    candidate: CompiledCandidate,
    tensors: tuple,
    workspace: torch.Tensor,
    out: torch.Tensor,
    reference_path: Path,
    reference: torch.Tensor,
) -> None:
    """Run both stages and enforce the same tolerance as remote_bench.py."""
    torch.cuda.synchronize()
    launch_full(candidate, tensors, workspace, out)
    torch.cuda.synchronize()

    reference_f32 = reference.float()
    diff = (out.float() - reference_f32).abs()
    bad = diff > (0.05 + 0.05 * reference_f32.abs())
    max_abs = float(diff.max())
    bad_count = int(bad.sum())
    total = bad.numel()
    print(
        f"correctness candidate={candidate.label} reference={reference_path} "
        f"max_abs={max_abs:.6f} bad={bad_count}/{total}",
        flush=True,
    )
    if not bad_count:
        return

    first_bad = torch.nonzero(bad, as_tuple=False)[0]
    row = int(first_bad[0])
    col = int(first_bad[1])
    raise RuntimeError(
        f"correctness failed for {candidate.path}: first_bad=({row},{col}) "
        f"got={float(out[row, col])} ref={float(reference[row, col])} "
        f"diff={float(diff[row, col])}"
    )


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
    candidate_paths = resolve_candidates(args.candidates)

    cfg, padded_total, tensors = rb.make_inputs(args.case, DEFAULT_SEED + args.case)
    expected_shape = (padded_total, cfg["hidden"])
    reference_path, reference = load_reference(args.case, expected_shape)

    workspace = torch.empty(
        (padded_total, cfg["intermediate"]),
        device="cuda",
        dtype=torch.float16,
    )
    out = torch.empty(expected_shape, device="cuda", dtype=torch.float16)

    candidates = compile_candidates(candidate_paths, cfg, padded_total, tensors)

    # Correctness always covers the complete Stage1 -> Stage2 path before any
    # performance numbers are produced.
    for candidate in candidates:
        check_correctness(
            candidate,
            tensors,
            workspace,
            out,
            reference_path,
            reference,
        )
    print("all_candidates_correct", flush=True)

    fixed_stage2_workspace = None
    if args.stage == "s2":
        # This buffer is written exactly once by candidate zero's Stage1 and is
        # thereafter passed unchanged to every candidate's Stage2.
        fixed_stage2_workspace = torch.empty_like(workspace)
        launch_stage1(candidates[0], tensors, fixed_stage2_workspace)
        torch.cuda.synchronize()
        print(
            f"fixed_stage2_workspace_source={candidates[0].label}",
            flush=True,
        )

    def selected_launch(candidate: CompiledCandidate) -> None:
        if args.stage == "s1":
            launch_stage1(candidate, tensors, workspace)
        elif args.stage == "s2":
            if fixed_stage2_workspace is None:
                raise RuntimeError("fixed Stage2 workspace was not initialized")
            launch_stage2(candidate, tensors, fixed_stage2_workspace, out)
        else:
            launch_full(candidate, tensors, workspace, out)

    for candidate in candidates:
        for _ in range(args.warmup):
            selected_launch(candidate)
        torch.cuda.synchronize()
        print(
            f"warmed candidate={candidate.label} launches={args.warmup}",
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
            f"round={round_index + 1}/{args.rounds} order={order_name}",
            flush=True,
        )
        for candidate in ordered:
            elapsed_ms = measure_events(
                lambda current=candidate: selected_launch(current),
                args.iters,
            )
            samples[candidate.index].append(elapsed_ms)
            print(
                f"timing round={round_index + 1} order={order_name} "
                f"stage={args.stage} candidate={candidate.label} "
                f"iters={args.iters} ms={elapsed_ms:.6f}",
                flush=True,
            )

    print_summary(candidates, samples)


if __name__ == "__main__":
    main()
