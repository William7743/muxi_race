# XPUOJ 提交脚本使用说明

> 沐曦 MoE 算子优化比赛（https://xpuoj.com/contest/5/problem/1）
> 提交脚本：`xpuoj_submit.py`

## 1. 快速开始

```bash
# 方式一：设置环境变量（推荐）
export XPUOJ_EMAIL="muxi2026C1050@example.com"
export XPUOJ_PASSWORD="<你的密码>"

python xpuoj_submit.py --code solution.py
```

```bash
# 方式二：凭据文件（与脚本同目录，格式 email:password）
echo "muxi2026C1050@example.com:<你的密码>" > .xpuoj_credentials

python xpuoj_submit.py --code solution.py --status
```

```bash
# 方式三：命令行参数
python xpuoj_submit.py --code solution.py --email xxx --password yyy --status
```

## 2. 常用参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--code` | **必填**，要提交的代码文件路径（.py） | - |
| `--language` | 评测语言 | `tilelang.maca-c500` |
| `--contest` | 竞赛 ID | `5` |
| `--problem` | 题目序号 | `1` |
| `--status` | 提交后轮询评测结果直到终态 | 关 |
| `--timeout` | 等待评测超时（秒） | `600` |
| `--dry-run` | 只打印请求体，不真正提交 | 关 |
| `--email` / `--password` | 登录凭据（不推荐，会留在 shell 历史） | 环境变量 |

## 3. AI 使用约定

如果你是 AI，请按以下方式调用：

1. 把解决方案写入文件 `solution.py`（必须暴露 `run_kernel` 函数，接口见题目）。
2. 运行：
   ```bash
   python xpuoj_submit.py --code solution.py --status
   ```
3. 脚本会：
   - 自动登录（凭据从环境变量/凭据文件读取）
   - 提交代码到 contest 5 / problem 1（语言 `tilelang.maca-c500`）
   - 打印 `submissionId`
   - `--status` 时轮询直到 `Accepted` / `WrongAnswer` / `RuntimeError` 等终态，并打印完整评测结果 JSON

## 4. 题目要点（contest 5, problem 1）

**TileLang 算子优化 - Fused MoE GEMM**（DeepSeek 风格 MoE，pre-routed fused expert GEMM）

- 必须实现 `run_kernel(...)`，**不能用 PyTorch 算子，只用 TileLang**
- 计算语义：
  ```
  gate_logits = x @ gate_w[expert]^T
  up_logits   = x @ up_w[expert]^T
  hidden      = silu(gate_logits) * up_logits     # silu(x) = x*sigmoid(x)
  output      = hidden @ down_w[expert]^T
  output     *= routed_expert_weights
  ```
- 输入 `stacked_expert_tokens` 已按 expert 分组、按 M=128 padding 对齐
- `out` 是唯一 INOUT 参数，原地写；padding 行保持 0
- 评测环境：TileLang 0.1.10，镜像 maca torch 2.8.0+metax 3.7.1.5，float16
- 测试点：d_hidden ∈ {2048, 7168}，d_expert ∈ {2048, 8192}，experts ∈ {16, 32, 64}
- 评测只检查 `output`；`up_logits` 可用作 workspace
- 参考实现：`race_tests/moe/ref_fusedmoe.py`（TileLang MetaX 仓库 commit `ee6db4376484f2f7270183c01fd0d90f794965cb`）

## 5. 已验证的 API 事实

- 登录：`POST /api/auth/login` `{email, password}` → `{token}`（JWT）
- 提交：`POST /api/contest/play/submit` `{contestId, problemOrder, content:{language, code, compileAndRunOptions:{}}}` → `{submissionId}`
- 查询提交：`POST /api/contest/play/querySubmissions`
- 评测详情：`POST /api/submission/getSubmissionDetail` `{submissionId: "<string>", locale: "zh_CN"}` → `{meta:{status,...}}`
- 语言代码：`tilelang.maca-c500`（不是 `tilelang`！）
- 评测终态：`Accepted` / `WrongAnswer` / `RuntimeError` / `CompileError` 等
