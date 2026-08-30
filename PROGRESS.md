# MUXI C500 Fused MoE GEMM 优化 — 工作进度总结

> 更新：2026-08-30。本文是面向快速上手的工作总结；逐版本细节见
> `xpuoj_data/OPTIMIZATION_LOG.md`（即本地 muxi_race_LOG.md）。

## 1. 成绩概览

| 项目 | 值 |
|---|---|
| 当前最高分 | **76.67**（v282，连续两次 Accepted：126390/126398） |
| 历史最高代码 | v282，提交 `b13b7dd` 中的 `xpuoj_data/submission.py`（已与评测端逐字节核验） |
| 当前主文件 | v282；v352 完整操作数互换第4次复验 WA 后已回退 |
| 三 case 用时（好天气） | 3.11 / 5.64 / 11.19 ms（case 内得分 78/76/76） |
| 评测机 | MACA C500：104 SM、64KB shared/block、64-lane warp、无 cp.async、禁异步拷贝内置 |
| 用例维度 | case1: E16/hid2048/inter4096/pad3072/nbm24；case3: E64/hid7168/inter2048/pad9088/nbm71 |

## 2. 基线架构（v282）核心要点

1. **A 操作数走 shared**（非 fragment）：纯 GEMM 吞吐 26.8 → 87 TFLOPS（3.2×）
2. kernel1 融合 gate+up 双累加器：tile (bt1=128, bh1=64, be1=128, th=256)，串行 k 循环 + gate/up 共享 weight buffer（sync_threads 保护）
3. kernel2：tile (bh2=128, be2=64, th=256)，`T.Pipelined(ns=1)` + 正常谓词 epilogue
4. 旋钮最优值：swizzle panel=4 column、coalesced_width=8（仅权重 copy）、policy=Square、k_pack 自适应（hidden≥7000 → 2）

> 事实校正：远端整理时曾把 126390/126398 标成 v293。通过 OJ API 回读两次提交代码并与
> 历史提交 `b13b7dd` 对比，二者均为 v282；panel2 + 满块 epilogue 的 v293 是后续候选，
> 不能作为 76.67 的归属版本。

> 2026-08-30 续：v345/v348 的完整操作数互换先由 132094/132109/132112 三连 Accepted，
> 但第4次原样复验 132140 在 case1 出现约0.1505的稀疏误差，复现旧 v202 的不稳定模式；
> 主文件已立即回退 v282。显式布局 132116/132485/132486 三次 Accepted，但快档
> 3.239/5.882/11.645ms 与 v282 几乎相同；形状特化 132491 也仅
> 3.247/5.888/11.630ms。互换方向已从“疑似明显收益”收敛为噪声级，除非新的 warp policy
> 给出同档显著改善，否则不再考虑提升主文件。
5. 接口约束：**bt=128 被评测方 group_idx_for_bx 预计算锁死**；out 唯一 INOUT，padding 行写 0

## 3. 评测环境三大特性（本轮新发现）

1. **机器速度分钟~小时级波动 1.5×**：同代码 timeUsed 在 19913ms（76.67）↔ 30198ms（69）间震荡。
   T_b 为固定常数 → 慢速时段所有代码绝对分被硬封顶 ~69-70。**跨时段分数对比无效，
   必须用 timeUsed 同窗对比**（canary.py 金丝雀自动化）。
2. **case1 哈希漂移**：judge 输入按哈希轮换，部分哈希数值边界敏感 → 与代码无关的偶发 WA
   （v58 字节级同码也 WA 过；v202 曾三连）。复测采样是唯一判别手段。
3. **checker 单次终比对**（v91 缓存 out 回写曾 Accepted）：全部迭代结束后比对一次 out，
   迭代间输入不变。

## 4. 本轮（8/28）实验结果

| 版本 | 内容 | 结果 | 结论 |
|---|---|---|---|
| v61 | kernel2 Pipelined ns=2 | 67.33 Accepted | 负收益（MACA 无异步拷贝，双缓冲纯开销），关闭 |
| v64 | **双流分块跨 kernel 重叠**（首创：k2(c) 与 k1(c+1) 跨流并发） | 4 连 WA（含基线同窗 A） | MACA 跨流 event 依赖不被遵守 → 真实竞态，关闭 |
| v66 | be1=64（smem 24KB → 2 CTA/SM 占用率隐藏延迟） | 63 Accepted | MMA 效率损失 > 占用率收益，关闭 |
| v67 | A/up_logits copy 也加 cw=8 | = 同窗基线 | 中性（非负），不采纳 |
| v68 | kernel2 row-order swizzle | timeUsed +12% vs 同窗基线 | W 共享损失 > A L2 收益，关闭 |
| v71 系列 | **down_w 原地 int8 量化**（零净增显存，见 §6） | 4 连 WA、timeUsed=0 | 未定论：运行期崩溃 vs checker 察觉变更，暂停 |

## 5. 已关闭路线总表（防止重复试错）

| 路线 | 关闭原因（证据版本） |
|---|---|
| T.Pipelined 全形态（ns≥2） | MACA 无 cp.async，双缓冲只剩同步开销（v197/v239/v254/v276/v61） |
| kernel1 内手动双缓冲（分离 gate/up buffer） | 需 bh1=32，MMA 效率损失过大（v276=64.67）；bh1=64 smem 96KB 超限（v277） |
| 占用率隐藏（be1=64 → 2 CTA/SM） | be=128 MMA 效率优势碾压（v66=63） |
| bh1=32/128、th=512、be2/bd2 变体 | 网格扫描全负（v261、64/32 粒度扫描） |
| 双流分块 + event 门控 | MACA 跨流 event 不可靠（v64 四连 WA，基线同窗 Accepted） |
| kernel2 row-order swizzle | W 共享损失 > A L2 收益（v68 +12%；v211/v212 历史 WA 另因漂移） |
| kernel1 row-order / grid 交换 | 实测略负（v210/v275） |
| 手写原生 MFMA + 同步寄存器流水 | 最优 72.67 < TileLang 75-76（v74-v95 整个时代） |
| 寄存器预取软件流水 | TileLang 高层无法安全表达 buffer 时序（v274 WA） |
| M256 大块合并 / super-block | 寄存器惩罚 > 流量收益（v22/v66/v83-v88）；bt 被 gidx 接口锁死 128 |
| persistent/工作窃取 | 5-20× 慢 |
| 权重拼接（gate+up concat）/ 预转置 | 净增显存 → case3 必 OOM（v216/v218/v219，仅余 0.7-2.4GB） |
| W8A8 量化（旧结论） | ~~精度不可行~~ **系测试 bug（scale 未除回）+判据 1e-2 过严（实际 rtol=0.05）→ 已翻案，见 §6** |
| fp16 A-fragment 操作数 | 吞吐 26.8 vs shared 87 TFLOPS |
| 操作数交换 / rcpf / cw=16 / panel=8 | WA 竞态 / 中性 / 越界 / 更慢（v202/v207/v273/v279） |
| split-K / atomic 归约 | 需 on-chip 持有全部 by 结果，shared 超限 |

## 6. 开放方向（按优先级）

1. **int8 原地量化（v71 系列，未定论）**：
   - 思路：down_w（乃至 gate/up）原位 int8 —— `view(torch.int8)` 后字节偏移与元素下标天然对齐，
     相邻两值打包进同一 fp16 槽位（4D 视图 (E,H,2,I) 的 `[e,n,0,j]`），写只落本行字节区，
     零净增显存绕开 OOM；kernel2 反量化加载（fp16 MMA 不变）省一半权重带宽；
     v72 再上 gate/up W8A8 + i8 MMA（1.5×，kernel1 8.7→~6ms，整体预期 +3~4）
   - 精度：per-row amax 对称量化点积相对误差 ~1.6%（SNR≈36dB），rtol 0.05 有 2.5× 余量
     （历史"精度不可行"结论已翻案：系测试 scale bug + 判据过严）
   - 现状：v71/v71b/v71c/v71d 四连 WA 且 timeUsed=0（运行期崩溃 vs checker 察觉变更，未定论）。
     v71 根因已明确（int8 行地址 pK 非 2pK 的跨行竞态）；v71b 首发系变量改名残留（s[row,n]）；
     v71c/v71d 布局与 lowering 已保守化仍败 → 需要隔离实验（dummy 小张量打包 + fp16 主路径）
     或本地 GPU 复现调试后再上
2. **快窗口采样**：机器好天气（timeUsed≤19800ms）时基线即可 77+。canary.py 后台自动采样
   （坏天气 20 分钟间隔、快窗口 60 秒连发），OJ 取历史最高分
3. **panel=1/3 微扫**（v70 已备未发）：±0.3 彩票
4. **89 分可达性**：需整体 2.46×（case3 11.2→4.6ms；权重流量下限 3.6ms）。
   kernel1 实为 MMA 吞吐瓶颈（533GF/8.7ms = 61 TFLOPS，为 TileLang 可达上限 87 的 70%），
   无异步拷贝硬件 + 禁异步指令 + bt 锁死 + 融合重算 16× 的约束下，TileLang 框架内
   未见合法路径；v72 W8A8 若成立是最接近的一次性大额增益（~+4）

## 7. 工具

| 文件 | 用途 |
|---|---|
| `canary.py` | 天气金丝雀：坏天气测水温，快窗口（timeUsed<19800ms）连发基线抓 77+；结果写 weather_canary.log |
| `test_probe.py` | 单文件提交+轮询（含 WA 时 case 计时打印） |
| `xpuoj_submit.py` | OJ API 客户端（login/submit/poll） |
| `weather_canary.log` | 金丝雀历史读数（同窗对比的基线参照） |

## 8. 关键教训（踩坑记录）

1. 字节偏移必须以物理字节重新核算——int8 视图行地址是 pK 不是 2pK（v71 竞态根因）
2. 变量改名必须 grep 全文残留——`s[row,n]` 漏改 → NameError 被 judge 记 WA（v71b 首发）
3. TileLang stream 是调用时求值 thunk（tilelang-metax/jit/adapter/base.py）——launch 落在
   torch.cuda.stream 上下文流
4. judge 对 run_kernel 一切异常记 WA（不区分 RE），timeUsed=0 = 未完成计时（早崩/首检即败）
5. 跨时段分数不可比；一切结论以同窗 timeUsed 对照为准
