#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 submission.py (v138) 派生两个诊断探针（不修改主文件）。

probe v196 (down-only): kernel1 grid 压缩为 1x1（仅 1 个空转块），
    kernel2 完整运行 → tk 时间 ≈ kernel2 单独耗时（结果必 WA：ws 为脏数据）。
probe v197 (stages2): 与 v138 完全一致，仅 kernel2 Pipelined num_stages=1 -> 2。
"""
import re
import pathlib

base = pathlib.Path(__file__).parent / "xpuoj_data"
src = (base / "submission.py").read_text()

# ---- probe v196: down-only ----
p1 = src.replace(
    "with T.Kernel(num_blocks_m, T.ceildiv(intermediate, be1), threads=th1) as (bx, by):",
    "with T.Kernel(1, 1, threads=th1) as (bx, by):  # PROBE: kernel1 disabled",
)
assert p1 != src
p1 = p1.replace(
    "# XPU-OJ v114: v106 with coalesced_width=4 on fused weight copies",
    "# XPU-OJ v196 PROBE: v138 with kernel1 grid collapsed to 1x1 (down-only timing).\n"
    "# 结果必 WA（up_logits 未写）；用 tk_time_ms 反推 kernel2 占比。",
)
(base / "probe_v196_down_only.py").write_text(p1)

# ---- probe v197: kernel2 num_stages=2 ----
p2 = src.replace(
    "for k in T.Pipelined(active_k_steps, num_stages=1):",
    "for k in T.Pipelined(active_k_steps, num_stages=2):  # PROBE: stages 1->2",
)
assert p2 != src
p2 = p2.replace(
    "# XPU-OJ v114: v106 with coalesced_width=4 on fused weight copies",
    "# XPU-OJ v197 PROBE: v138 with kernel2 Pipelined num_stages=1 -> 2。\n"
    "# 测试 down GEMM 双缓冲是否有收益（smem 32KB，双缓冲仍 <=64KB）。",
)
(base / "probe_v197_down_stages2.py").write_text(p2)

print("generated:",
      (base / "probe_v196_down_only.py").stat().st_size,
      (base / "probe_v197_down_stages2.py").stat().st_size)
