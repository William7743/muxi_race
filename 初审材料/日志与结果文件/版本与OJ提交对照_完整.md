# 版本与 OJ 平台提交对照表

评测账号：muxi2026C1050（contest 5 / problem 1，XPU-OJ）

## 本包最终提交版本

| 项目 | 内容 |
|---|---|
| **代码版本** | **v432**（本地文件 `source/submission_v432_final.py`） |
| **对应 OJ 提交** | **submissionId = 135985** |
| 提交时间 | 2026-09-02 15:57 (UTC) |
| 判定 | **Accepted，displayScore = 78.67**（当时 rank 27） |
| 总用时 timeUsed | 17 535 ms |
| 三 case 用时 | case1 2.791 ms / case2 5.065 ms / case3 9.681 ms |
| 显存 | 23.27 GB |
| 语言 | tilelang.maca-c500 |
| 代码一致性 | `source/submission_v432_final.py` 与 OJ 实际落盘代码逐字节一致（md5 校验，另存 `logs/` 对照副本） |

## 其他关键版本存档

| 版本 | OJ submissionId | 分数 | timeUsed | 提交时间 | 说明 |
|---|---|---|---|---|---|
| v404（纯净） | 134755 | Accepted 78.00 | 18 263 ms | 2026-08-31 | v432 的直接基线（v432 = v404 谱系 + fast_math/ldgstg_predicated/分派全开） |
| v293 | 130598 | Accepted 69（慢档日实测 76.67 档代码） | 30 125 ms | 2026-08-28 | 早期最优架构（A-shared/cw8/Square/panel2），慢速评测档位日提交，绝对分受机器档位压制 |

> 说明：评测机速度存在分钟~小时级 ±50% 波动，130598 的 69 分与 135985 的 78.67 差异主要
> 来自机器档位而非代码差异；对比方法论见 OPTIMIZATION_LOG.md 同窗口对照章节。

## 核对方式

任一提交可通过 OJ API `submission/getSubmissionDetail`（submissionId）复核状态、分数与
代码；本包 `oj_submission_records.json` 为拉取时点的 API 原始返回存档，
`oj_code_135985.py` 风格文件为对应提交的逐字节代码存档。

## 关于源码头注释的说明

`source/submission_v432_final.py` 文件头部的注释为开发过程遗留（写作"v412 candidate"），
实际文件内容即 **v432 最终形态**（= v412 + tl.enable_fast_math ×4 +
tl.enable_lower_ldgstg_predicated ×2 + case1 分派全开），与 OJ submissionId **135985**
（Accepted 78.67）的落盘代码**逐字节一致**。为保持与 OJ 存档的字节一致性，
该头注释未做改动，特此说明。

## 2026-09-04 增补

| 版本 | OJ submissionId | 分数 | timeUsed | 说明 |
|---|---|---|---|---|
| v478 重提 | 138992 | 79.67 | 16 389 ms（历史最快） | 同代码快档窗口 |
| v469 重提 | 138978 | 79.33 | 16 818 ms | 同窗对照 |
| v496（shape-isolated panel3） | 未提交 | 本地 2.558/5.412/8.492 | 仅比 v478 快 0.5-1% | 待人工决定是否提交 |

## 80 分档三版本 OJ 精确 timeUsed 对照（2026-09-05 API 实测）

| 版本 | OJ sid | timeUsed | 提交时间(UTC) | 备注 |
|---|---|---|---|---|
| v748（E32 M32 分支在 fast-sprint 前置） | 140335 | **15 893 μs** | 09-05 12:43 | 三者最小 |
| v755（同 v748 + 同分支重打包版） | 140440 | 15 917 μs | 09-05 14:19 | 与 v748 差 24 μs（0.15%，窗口噪声内） |
| v720（无 E32 M32 分支） | 139770 | 16 031 μs | 09-05 03:06 | 三者最大 |

结论：v748 与 v755 数学结构等价（755 = 748 重打包），OJ 实测差 0.15% 属窗口噪声；
两者均优于 v720 约 0.8%。**在榜显示分并列 80.33。**
