# Fused MoE GEMM（TileLang on MetaX C500）— 源码与测试说明

## 1. 文件清单

| 文件 | 说明 |
|---|---|
| `submission_v755_final.py` | **最终提交版本**（OJ 提交号 140440，Accepted **80.33**，2026-09-05，timeUsed 15 917 μs 历史最快；SHA256 0b19e84d… 与 GPT5.6 终版记录一致）。包含全部 TileLang kernel 与 `run_kernel` 入口 |
| `remote_bench.py` | 判题器同构的本地正确性/性能测试 harness（单进程单 case，避免 JIT 符号缓存串扰） |

## 2. 环境依赖

| 组件 | 版本（评测/验证环境实测） |
|---|---|
| GPU | 沐曦 MetaX C500（mx-smi 驱动 3.8.30 实测通过） |
| MACA 软件栈 | 3.7.1 / 3.8.x |
| Python | 3.12（conda） |
| PyTorch | 2.8.0+metax3.7.1.3 |
| TileLang | tilelang-metax 0.1.9/0.1.10（race 分支，commit ee6db43 及之后） |

> 仅需推理评测，无需训练环境；无其他第三方依赖（`torch`、`tilelang` 之外仅标准库）。

## 3. 运行方式

### 3.1 评测平台（XPU-OJ）

OJ 自动完成编译与调用：评测程序 import 提交文件后，按题目接口调用
`run_kernel(stacked_expert_tokens, gate_w, up_w, down_w, routed_expert_weights,
group_sizes, group_offsets, group_padded_offsets, group_idx_for_bx, out)`。
无需任何额外操作。

### 3.2 本地功能测试（正确性）

```bash
# 1) 用可信版本（OJ Accepted 的 v432）生成参照输出
python remote_bench.py --case 1 --candidate submission_v432_final.py \
       --warmup 5 --iters 20 --save-output ref_case1.pt
# 2) 用候选版本对拍（fresh process，避免 JIT 缓存串扰）
python remote_bench.py --case 1 --candidate <候选文件>.py \
       --reference-output ref_case1.pt --warmup 5 --iters 20
```

`--case` 支持 0/1/2/3/4/5/6：其中 1/2/3 为 OJ 三个正式 case
（E×hidden×inter×valid = 16×2048×8192×2272、32×7168×2048×4544、64×7168×2048×9088），
0/4/5/6 为边界形状（单 expert、valid=127 非整块、hidden=128 等）。
正确性判定输出 `max_abs` 与 `bad`（超出 rtol=0.05/atol=0.05 的元素个数）。

### 3.3 本地性能测试（benchmark）

```bash
python remote_bench.py --case 2 --candidate submission_v432_final.py \
       --warmup 10 --iters 100
# 输出 candidate_ms（CUDA event 计时，warmup 后取平均）
```

另支持 `--phase stage1|stage2` 对两级 kernel 分别计时（要求候选文件含
`_get_stage1` / `_get_stage2` 工厂函数，`submission_v432_final.py` 满足）。

## 4. 实测结果摘要（MetaX C500 实测）

| 用例 | 本地 mean (warmup10/iters100) | OJ mean（submissionId 135985） |
|---|---|---|
| case1 (E16, 2048×8192, valid 2272) | 2.544 ms | —（同窗 v719 2.563） |
| case2 (E32, 7168×2048, valid 4544) | 4.616 ms | 4.600 ms |
| case3 (E64, 7168×2048, valid 9088) | ~9 ms（页面舍入） | 8.926 ms（v718） |
| **OJ 总分** | | **80.33（Accepted，2026-09-05；三点 81/80/80）** |
| OJ 总用时 | 15 917 μs（历史最快） |

正确性：三点全量对拍（对官方 fp32 参考语义与 OJ Accepted 参照）bad=0；含 NaN 污染三轮复算、双 dtype 路由权重、两次新输入调用覆盖；
另通过 32 组随机 seed 认证（20×case1 + 6×case2 + 6×case3）零失败。

版本谱系：**v755（140440，80.33，15 917 μs）= 冻结 v748 + E32 Stage1 ≤32 行 M32 尾块分支**（尾块 padding 从 ~100 行降到 ~4 行）；v748 谱系含 v469/v470/v471/v478/v496 全部要素（双侧 vec4 布局/cw 匹配/panel2/k_pack=2/手写同步调度/E64 terminal-K/E16+E64 双B emitter）。三份独立代码（v718/v719/v755）均 80.33，成绩可复现。详细技术方案见提交包根目录《技术方案文档》。
