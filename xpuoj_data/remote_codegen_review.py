"""Compile a selected MoE stage and report generated source without launching it.

Run with the contest TileLang runtime and remote_bench on sys.path. This is a
diagnostic, never part of a submission or its measured run_kernel path.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import tilelang.language as T

import remote_bench as rb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--padded-tokens", type=int, required=True)
    parser.add_argument("--valid-tokens", type=int, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--weights-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--full-source", action="store_true")
    parser.add_argument("--candidates", nargs="+", required=True)
    args = parser.parse_args()
    if min(args.padded_tokens, args.valid_tokens, args.blocks) < 0:
        parser.error("metadata dimensions must be nonnegative")
    cfg = rb.CASES[args.case]
    for index, filename in enumerate(args.candidates):
        path = Path(filename).resolve(strict=True)
        module = rb.load_submission(
            str(path), f"codegen_review_{index}", kernel_suffix=f"codegen_review_{index}"
        )
        if args.stage == 1:
            kernel = module._get_stage1(
                cfg["hidden"], cfg["intermediate"], cfg["experts"],
                args.padded_tokens, args.blocks,
            )
        else:
            kernel = module._get_stage2(
                cfg["hidden"], cfg["intermediate"], cfg["experts"],
                args.padded_tokens, args.valid_tokens, args.blocks,
                T.float16 if args.weights_dtype == "float16" else T.float32,
            )
        source = kernel.get_kernel_source()
        lines = source.splitlines()
        summary = {
            "file": path.name,
            "stage": args.stage,
            "case": args.case,
            "weights_dtype": args.weights_dtype,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_characters": len(source),
            "static_syncthreads_sites": source.count("__syncthreads()"),
            "shared_offsets": [s.strip() for s in lines if "void* " in s and "buf_dyn_shmem" in s],
            "local_array_declarations": [
                s.strip() for s in lines
                if re.match(r"\s*(?:float|half_t|int)\s+\w+\[\d+\];", s)
            ],
            "outer_loop_headers": [
                s.strip() for s in lines if re.search(r"for \(int (?:k|outer)\b", s)
            ],
            "route_load_source_lines": [
                s.strip() for s in lines if "routed_expert_weights[" in s
            ],
            "note": "Static source sites/arrays are not dynamic barrier counts or physical register usage.",
        }
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.full_source:
            print(f"SOURCE_BEGIN {path.name}", flush=True)
            print(source, flush=True)
            print(f"SOURCE_END {path.name}", flush=True)


if __name__ == "__main__":
    main()
