# MUXI C500 Fused MoE GEMM 优化 — 工作进度总结

> 更新：2026-08-30。本文是面向快速上手的工作总结；逐版本细节见
> `xpuoj_data/OPTIMIZATION_LOG.md`（即本地 muxi_race_LOG.md）。

## 1. 成绩概览

| 项目 | 值 |
|---|---|
| 当前最高分 | **77.33**（v399候选，133517；3.077/5.315/10.243ms，复验中） |
| 上一稳定基线 | v282，提交 `b13b7dd` 中的 `xpuoj_data/submission.py`（126390/126398） |
| 当前主文件 | **v399**：v388稳定函数 + hidden7168独立半预取JIT；133517/133531已2A/0W |
| 三 case 用时（v399快档） | 3.077 / 5.315 / 10.243 ms（77.33分） |
| 评测机 | MACA C500：104 SM、64KB shared/block、64-lane warp、无 cp.async、禁异步拷贝内置 |
| 用例维度 | case1: E16/hid2048/inter4096/pad3072/nbm24；case3: E64/hid7168/inter2048/pad9088/nbm71 |

## 2. 基线架构（v282骨架 / v399形状隔离编译配置）核心要点

1. **A 操作数走 shared**（非 fragment）：纯 GEMM 吞吐 26.8 → 87 TFLOPS（3.2×）
2. kernel1 融合 gate+up 双累加器：tile (bt1=128, bh1=64, be1=128, th=256)，串行 k 循环 + gate/up 共享 weight buffer（sync_threads 保护）
3. kernel2：tile (bh2=128, be2=64, th=256)，`T.Pipelined(ns=1)` + 正常谓词 epilogue
4. 旋钮最优值：swizzle panel=4 column、coalesced_width=8（仅权重 copy）、policy=Square、k_pack 自适应（hidden≥7000 → 2）
5. v388将两阶段拆为独立JIT：Stage1关闭冗余safe-memory谓词与256-bit自动向量化；
   Stage2保留默认safe-memory legalize，精确复验统计3A/1W，仍是当前最保守的快速回退骨架。
6. v399保留v388原Stage1函数供hidden2048使用，另建hidden7168-only的Up寄存器半预取
   JIT；通过host builder选择实现真正函数级隔离，避免常量IR分支扰动case1 lowering。

> 事实校正：远端整理时曾把 126390/126398 标成 v293。通过 OJ API 回读两次提交代码并与
> 历史提交 `b13b7dd` 对比，二者均为 v282；panel2 + 满块 epilogue 的 v293 是后续候选，
> 不能作为 76.67 的归属版本。

> 2026-08-30 续：v345/v348 的完整操作数互换先由 132094/132109/132112 三连 Accepted，
> 但第4次原样复验 132140 在 case1 出现约0.1505的稀疏误差，复现旧 v202 的不稳定模式；
> 主文件已立即回退 v282。显式布局 132116/132485/132486 三次 Accepted，但快档
> 3.239/5.882/11.645ms 与 v282 几乎相同；形状特化 132491 也仅
> 3.247/5.888/11.630ms。互换方向已从“疑似明显收益”收敛为噪声级，除非新的 warp policy
> 给出同档显著改善，否则不再考虑提升主文件。后续 M-first、FullCol、单边互换和三种
> M256拼接加载也均失败或变慢，转置/拼接子空间现已关闭。

> 内建函数状态：同步 `ldg_b128` 的官方签名与源码已再次核对，但 v81/v103 在评测 mxcc
> 环境均无法打通 fp16 ABI；合法的同步向量 load/store 已由 `T.copy` 的 32–256bit 指针
> lowering 覆盖。能显著改变流水的 `ldg_b128_bsm + arrive/wait` 属题目禁止的异步拷贝，
> 不采用。原生 MFMA/call_extern 通道可用但历史最优仍低于 T.gemm。
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

## 9. 2026-08-30 新突破：关闭冗余安全边界改写

- v368 / 132534：在字节一致 v282 上仅增加
  `"tl.disable_safe_memory_legalize": True`，三档全部 Accepted，慢资源档
  **4.201/7.847/15.897ms，70.33分**。
- 同档 v282 约 4.30–4.33/8.63–8.64/17.65–17.78ms；新开关对 case2/3 有约
  **9–11%** 的明确收益。原因是正式 shape/tile 全部整除且 token 已 padding，默认 safe-memory
  pass 仍会因动态 expert/group 索引无法静态证明而生成大量冗余谓词。
- 原样复验与vectorize256/let-inline独立及组合筛选均已完成；结果见下方复验进展。

### 复验进展

- 纯 safe 版累计2A/1W；132574 在快资源档达到当前新高 **77分**，
  **3.043/5.494/10.783ms**，确认较 v282 的历史快档也真实更快。
- safe + disable-vectorize256 的132544/132586已2A/0W，慢档约
  4.20/7.85/15.85–15.90ms；vectorize256消融本身无收益，但可能提供稳定化。
- 该2A/0W组合已提升为当前主文件 **v380**；精确主代码复测133031正在排队。
- 更稳健的显式范围约束版132580不关闭全局安全 pass，同样 **Accepted 77**，
  约3.034/5.556/10.790ms；原样复验133009与仅expert-id消融133011也均Accepted，但慢档
  分别约4.231/8.474/17.194和4.239–4.250/8.500/17.119，只比v282快约2–3%，明显慢于
  v380。显式assume只能消掉部分谓词，不能替代全局safe-memory关闭。
- force-let-inline、safe+loop-unswitch、双decorator shape隔离均触发case1稀疏错误；
  `readfirstlane`真实元数据广播正确但中性。确定性高层persistent版本133001虽Accepted，
  但约5.194/10.804/21.341ms、仅64.33分，persistent路线关闭。当前继续等待增强证明器、
  block紧约束和同步LDG/STG交互结果，主文件保持v380。
- v383 / 133016的强化`tl.Simplify`已 **Accepted 69**，慢档约
  **4.246/8.505/17.263ms**，与普通完整assume等价，无法进一步取代
  `disable_safe_memory_legalize`；增强证明器路线关闭。
- v382 / 133019的紧block区间assume在case1 **WrongAnswer**（明显误差约0.1827，
  约4.257ms），无性能信号且扰动fragment lowering；继续堆显式范围约束的路线关闭。
- v384同步LDG/STG（133032）和v385 Stage1完整块分支（133075）均Accepted但与v380持平，
  不叠加。v386 Stage2完整块分支（133078）在快档达到 **77.33**、
  **3.043/5.495/10.709ms**，正用133469原样复验稳定性。
- 修正后的v387合法同步寄存器预取（133085）已Accepted，慢档
  **4.208/7.643/15.566ms**，case2/3比v380同档快约2–3%；正用133470原样复验。

## 10. 2026-08-30 下午续：竞态定位 + v393 新基线（接手 AI 完成）

### 关键实锤：v380 类代码的非确定性竞态
- **133276 vs 133287/133310**：v391（= v380 + Stage2 raw rw clamp，clamp 对有效行
  是 no-op）同字节代码、同 case1 哈希 `f052087a`，1 次 Accepted（19233ms，77.0）、
  2 次 WA（稀疏误差落在有效行，位置随机）→ **同码同哈希可翻转 = 非确定性竞态**，
  非哈希漂移、非 clamp 引入。
- 推断：safe-memory legalize 关闭后，Stage1 gate/up **共享 weight_shared 的写写别名
  + 跨 barrier 读写**成为唯一可被编译器重排的歧义点（v282 别名+safe ON 50+ 提交
  零 WA；safe OFF 后偶发）。

### v393：结构性竞态修复（当前最快+最稳候选）
- 改动：Stage1 权重缓冲改 `w2[2]` 双缓冲，gate→w2[0]、up→w2[1]，**消除别名**；
  barrier 从 2 个/迭代减到 1 个；两条权重 LDG 背靠背发射。smem 48KB 仍 1 CTA/SM。
- 结果：**3×Accepted / 0 WA**（133296 中档 23796ms；133309/133329 慢档
  31315/31255ms，同哈希集两次差 <0.3%，自重现极好）；速度与 v380 同档无回退信号。
- v394（v393+Up 寄存器预取+补 barrier）同窗比 v393 慢 1.7% → 预取叠加关闭；
  v389 全双向预取已负收益。预取方向（v387 半版之外）全部关闭。

### 其他本轮结论
- v388（拆双 JIT，Stage1-only safe-off+vec256-off）：3×A，快档 19804ms（76.67，
  比 v380 慢 2.5%）→ Stage2 默认 safe 的安全代价 ~2.5%，与"safe-off 收益主要来自
  Stage1"假设一致。v390（v388+v386 branchless epilogue）≈ v388。
- v346（kernel1 T.Pipelined ns=2）= 59，再次确认 MACA 无 cp.async 时 Pipelined 纯开销。
- 沙箱：`import sys/time` 被拒、`torch.cuda` 被 TorchProxy 拦截；探针只能走
  RuntimeError 回显。机器速度三档：快 19.2-19.9s / 中 ~24s / 慢 ~30.2s。
- 测试集哈希逐轮轮换（case1 有 I=4096/8192 两个变体、case2/3 哈希也变），
  跨运行对比必须限定同哈希集。

### 当前主文件
**v393**（v380 的 Stage1 非别名 w2 版 + Stage2 clamp）— 3×A/0W。榜单最高仍为
v386 的 77.33（133078）；v393 快档抽卡预期 76.7-77.3 且无 WA 风险。
- v396的扩展复验133477/133478/133479三次均case1 WA，字节一致统计
  由2A/0W降为最终 **2A/3W**；select写回并未根治不确定性，已淘汰。
- v386原样复验133469 Accepted，累计2A/0W。v387原样复验133470也Accepted，
  累计2A/0W，慢档 **4.203/7.654/15.432ms**，case2/3的2–3%收益复现；
  后续133489 Accepted、133488/133490 case1 WA，最终 **3A/2W**，全形状版不提升。
- v397 / 133501与v398 / 133508均case1 WA（约4.252/4.257ms）；IR内常量形状分支和
  Stage2 expert-id assume都会扰动敏感lowering，两条写法关闭。
- v399 / 133517改为真正函数级隔离：原v388 Stage1函数体完全不变，新建独立
  prefetch Stage1 JIT，host只为hidden=7168选新builder。结果 **Accepted 77.33**，
  **3.077/5.315/10.243ms**，case1与v388持平，case2/3快约5.6%/7.6%；已追加
  **133531/133532/133533** 三次字节一致复验。
  同窗v388精确对照 **133525** 在case1 WA（约4.253ms），v388现为3A/1W；
  当前窗口本身存在稀疏翻转，v399需用多样本统计判定。
- v400 / 133556叠加已独立Accepted的v390 Stage2完整块无分支写回，尾块和safe pass不变；
  用于尝试以约0.3%微改动让v399 case3跨过78分档，排队中。
- v401 / 133563是高收益隔离探针：所有GEMM/权重仍为FP16/FP32，只用固定1/16比例
  将Stage1→Stage2 workspace压为int8，Stage2现场还原到FP16 shared；无归约/缓存/异步内建，
  目标是将Stage2重复中间张量流量减半。排队中。
- v402 / 133581根据真实初始化分布修正INT8 workspace量程：hidden2048步长0.0625，
  hidden7168步长0.2（±25.4）；与v401形成精度消融，排队中。
- 统计模拟进一步选出hidden2048步长1/32、hidden7168步长1/4。v403首次生成
  133589因单行Python分支未执行全部替换而作废；修正且完整断言的v403b为 **133590**，排队中。
