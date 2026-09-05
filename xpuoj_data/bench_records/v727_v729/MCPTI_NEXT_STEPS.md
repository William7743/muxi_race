# MCPTI：最低风险的 Stage1 计数器验证清单

2026-09-05，独立文件复核与主线程元数据实跑结果归档。**已完成 event3 的 add/query/destroy，但尚未启用计数器或读取计数值**；没有 Stage1 采集结果、带宽或利用率结论。本次文档/CPU测试作者未执行 GPU、SSH 或修改提交候选。

最新证据：[event3 preflight 原始 JSON](codex_mcpti_event3_preflight_20260905.json)。`mcptiGetVersion` 返回18；初始化、context、create、add、11项属性查询及最终 destroy 的状态均为0。event 名称/短描述/长描述的13字节原值均为 `6163746976655f776172707300`（ASCII `active_warps` 加末尾 NUL）。这已经比此前 empty-group-only 验证多证实了 **event3 可以加入组，且属性可读**。

event scope 与 group scope 均返回4字节 `01000000`；group all-instances 为 `01000000`，instance-count 为 `0d000000`。这里只转录原始bytes，**不据此认定已限制 context-only、默认实例范围安全或恰好采集13个硬件单元**。group/domain 属性的 payload 契约和可写性仍未获得充分声明；尤其不能将 `all_instances_changed=false` 误读为默认 all-instances 已关闭。结果明确为 `metadata_only`，`context_scope_verified=false, collection_enable_attempted=false, counter_values=null`，warmup/采集 launch 均为0。

## 1. 目前已经证实与尚未证实的内容

复核源为 [`remote_mcpti_inventory.py`](../../remote_mcpti_inventory.py)、初始 [`codex_mcpti_inventory_20260905.json`](codex_mcpti_inventory_20260905.json)，以及新取得的 [`codex_mcpti_inventory_group_20260905.json`](../v730/codex_mcpti_inventory_group_20260905.json)。安装头文件已下载在仓库上一级 `mcpti/mcpti.h` 与 `mcpti/mcpti_type.h`，已独立读取相关声明、枚举和注释。

- JSON 记录 device 0、`/opt/maca-3.7.1`，报告数量和实际列表均为 166；166 个 `event_count_status` 都为 0。
- 166 个 `description` 都与 `name` 相等，没有事件定义、单位、公式、采样范围或百分比分母信息。
- 初始 JSON 只有 inventory；group-smoke JSON 已证实空组创建/销毁成功：`context_status=0, has_context=true, create_status=0, has_group=true, destroy_status=0, collection_enabled=false`。**该次**尚未添加事件；之后的 event3 preflight 已完成 add/query/destroy，见上方最新JSON。两次均未启用采集。
- 枚举成功不证明事件组可配置、计数器资源可分配、当前切片有采集权限、read 返回有效数据，或 metric 求值成功。
- 已核对头文件 SHA256：`mcpti.h = 8d280365099be08c332c2960fb7ef71314274746acfad799c001b94229fc8b1b`；`mcpti_type.h = a9be1b75203e1cee64b2f453fcb6e2f5496fb188116f0d55e510e86ab7e32f69`。下文数值与原型来自这两份安装头文件，不是推测 CUPTI 的相似接口。`MCdevice/MCcontext` 所依赖的 `maca.h`、runtime context/sync 声明未包含在这两份文件内，仍不能声称所有依赖 typedef 均已独立核验。

## 2. 已见绑定与不能外推的 ABI

下表是当前脚本已用绑定。`mcpti_type.h:53-56` 明确 metric/event/domain ID 为 `uint32_t`，event group 为 `void*`；`MCptiResult` 是含 `FORCE_INT` 的枚举，成功值明确为 0。Linux 下 `MCPTIAPI` 为空，头文件提供 C 链接声明；目标 Linux 进程仍须检查 ctypes 基础类型尺寸，不能采用本机 Windows 的 `long` 宽度代替目标 `size_t`。

| 接口 | 脚本当前参数类型 | 证据边界 |
| --- | --- | --- |
| `mcSetDevice` | `c_int` | 脚本启动步骤；不能单凭 metric 枚举证明已取得有效 current context |
| `mcptiDeviceGetNumMetrics` | `c_int, uint32_t*` | 本 JSON 对应枚举流程成功 |
| `mcptiDeviceEnumMetrics` | `c_int, size_t*, uint32_t*` | 脚本把容量/返回大小作为字节数，并检查越界与整除 |
| `mcptiMetricGetAttribute` | `uint32_t, c_int, size_t*, void*` | 头文件确认 name=0、short description=1；新增属性见下文核验表 |
| `mcptiMetricGetNumEvents` | `uint32_t, uint32_t*` | 166 项返回成功 |
| `mcptiMetricEnumEvents` | `uint32_t, size_t*, uint32_t*` | 当前 JSON 中所列 event IDs 来自此流程，不是新定义的枚举常量 |
| `mcCtxGetCurrent` | `void**` | group-smoke 已返回成功且 context 非空；runtime 原始声明仍需保留其依赖头文件 |
| `mcptiEventGroupCreate` | `MCcontext, MCpti_EventGroup*, uint32_t` | `mcpti.h:1616`；flags 明确要求 0；group-smoke 创建成功 |
| `mcptiEventGroupDestroy` | `MCpti_EventGroup` | `mcpti.h:1633`；group-smoke 销毁成功；头文件禁止销毁 enabled group |

### 2.1 已核验的最小采集函数原型

以下均返回 `MCptiResult`，枚举参数按声明的枚举处理；`size_t*` 不能写成 `uint32_t*`。

| 函数与头文件行号 | 参数列表 |
| --- | --- |
| `mcptiSetEventCollectionMode`，1489 | `MCcontext, MCpti_EventCollectionMode` |
| `mcptiEventGetAttribute`，1181 | `MCpti_EventID, MCpti_EventAttribute, size_t*, void*` |
| `mcptiEventGroupAddEvent`，1560 | `MCpti_EventGroup, MCpti_EventID` |
| `mcptiEventGroupGetAttribute`，1230 | `MCpti_EventGroup, MCpti_EventGroupAttribute, size_t*, void*` |
| `mcptiEventGroupSetAttribute`，1375 | `MCpti_EventGroup, MCpti_EventGroupAttribute, size_t, void*` |
| `mcptiDeviceGetEventDomainAttribute`，1102 | `MCdevice, MCpti_EventDomainID, MCpti_EventDomainAttribute, size_t*, void*` |
| `mcptiEventDomainGetAttribute`，1150 | `MCpti_EventDomainID, MCpti_EventDomainAttribute, size_t*, void*` |
| `mcptiEventGroupEnable/Disable/ResetAllEvents`，1029/1045/1655 | 各为 `MCpti_EventGroup` |
| `mcptiEventGroupReadEvent`，1351 | `MCpti_EventGroup, MCpti_ReadEventFlags, MCpti_EventID, size_t*, uint64_t*` |
| `mcptiEventGroupReadAllEvents`，1299 | `MCpti_EventGroup, MCpti_ReadEventFlags, size_t*, uint64_t*, size_t*, MCpti_EventID*, size_t*` |

`ReadAllEvents` 最后一个参数 `numEventIdsRead` 确实为 `size_t*`。`ReadEvent` 最简单；只读 event3 时不必实现多事件数组与 group-set struct。

### 2.2 已核验常量、读数布局及顺序

来自 `mcpti_type.h:11584-11650`：

- collection mode：`CONTINUOUS=0, KERNEL=1`；read flag：`NONE=0`。
- event profiling scope：`CONTEXT=0, DEVICE=1, BOTH=2`。
- event 属性：`NAME=0, SHORT_DESCRIPTION=1, LONG_DESCRIPTION=2, CATEGORY=3, PROFILING_SCOPE=5`；不能误造不存在的通用 event 单位属性。
- group 属性：`EVENT_DOMAIN_ID=0, PROFILE_ALL_DOMAIN_INSTANCES=1, USER_DATA=2, NUM_EVENTS=3, EVENTS=4, INSTANCE_COUNT=5, PROFILING_SCOPE=6`。
- domain 属性：`NAME=0, INSTANCE_COUNT=1, TOTAL_INSTANCE_COUNT=3, COLLECTION_METHOD=4`。
- `MCPTI_EVENT_OVERFLOW=0xffffffffffffffff`，`MCPTI_EVENT_INVALID=0xfffffffffffffffe`，均为 `uint64_t`；不能把它们当巨大的有效计数。

`mcpti.h:1014-1045,1235-1355` 明确：**Enable 自动清零并开始收集；Read 必须在 group 仍 enabled 时调用，读后清零；Disable 停止收集。** 所以单次顺序必须是 `enable → Stage1 → synchronize → read → disable → destroy`。不需要在 Enable 后再并发 reset；Read 与 reset 不可并发。

单事件默认未启用 all-instances 时，value buffer 至少 `sizeof(uint64_t)`；开启 all-instances 才要求乘 domain instance 数。首版不主动开启 all-instances，不请求 device-wide 范围。`ReadAllEvents` 布局为 **instance-major**：每个 instance 内依次排列 event0…eventN，event 顺序由返回的 `eventIdArray` 决定；不能假设就是输入顺序。所有容量与返回长度为字节数，返回事件个数另用 `numEventIdsRead`。

### 2.3 尚存的 typed-attribute 证据缺口：保守中止条件

头文件给出了通用 `Get/SetAttribute(..., void*)` 和上述属性ID，但 **group/domain 枚举没有逐项说明 payload 是哪种整数类型/宽度、哪些属性可写**。这与明确写了 value-type 的 metric 属性不同。因此不能仅因“多数 CUPTI 示例用 uint32_t”就强行写 scope/all-instances/count。查询可以先用有上限的原始 byte buffer，记录状态、实际长度和原始字节；不要把一段4字节返回值自动当成已证明的属性类型。

当前约束要求 mode=KERNEL、scope=CONTEXT 显式设置并验证；如果缺乏属性 payload 类型证据，或 scope setter/readback 不支持，则**在 enable 前中止，只输出 metadata**。事件自身返回 BOTH 也不等于 group 已限制为 CONTEXT。安装头文件有 `mcptiSetEventCollectionMode`，但未找到对称的 current-mode getter；setter 返回0只能标作设置成功，不能捏造 mode readback。若要求必须严格读回 mode，当前公开声明不够满足。

不使用 `mcptiDisableKernelReplayMode` 充当无副作用的状态查询：它会改变 mode 并禁用既有事件组。首版独立进程不调用任何 replay API。

### 2.4 延后求值与多 pass：ABI 已见，首版不用

- `mcpti_type.h:11765-11785` 确认 metric `VALUE_KIND=4, EVALUATION_MODE=5`；value-kind 为 DOUBLE=0、UINT64=1、PERCENT=2、THROUGHPUT=3、INT64=4、UTILIZATION_LEVEL=5。union 字段分别是 double/uint64/int64/double/uint64/枚举；不要将所有结果按 double 解码。目标 ABI 的 union 尺寸/对齐仍应做实际编译期检查。
- `mcptiMetricGetValue`（`mcpti.h:2271`）参数为 `MCdevice, MCpti_MetricID, size_t, MCpti_EventID*, size_t, uint64_t*, uint64_t, MCpti_MetricValue*`，其中 duration 的头文件单位明确为 **ns**，event/value 按对应顺序传入。THROUGHPUT 的 union 文档单位为 bytes/second；这不是本轮已经测得的带宽。
- aggregate 求值有厂商定义的 instance 归一化要求；PER_INSTANCE=1、AGGREGATE=2 是 bit flags。首版保存原始数组，不执行归一化、不乘4外推切片。
- `mcptiEventGroupSetsCreate`（1452）参数为 `MCcontext, size_t, MCpti_EventID*, MCpti_EventGroupSets**`，可报告所需 pass；`mcpti_type.h:11685-11693` 的 group set 为 `uint32_t numEventGroups; MCpti_EventGroup* eventGroups`，sets 为 `uint32_t numSets; MCpti_EventGroupSet* sets`。不要手动 packed；首版单事件不需要它。

实现宜有超时和 `try/finally`：仅对成功取得的非空 handle 操作；记录每步状态；已 enable 才尝试 disable；成功创建的 group 最终释放。不要重置设备、停止他人进程或更改驱动来让 smoke 成功。

## 3. 最小事件集合：先单项通路，再最小诊断

当前已落盘的工具是 [`remote_mcpti_event_preflight.py`](../../remote_mcpti_event_preflight.py)，**只做 metadata preflight，不是 Stage1 collector**。它不导入 torch、不编译候选、不分配模型数据、不调用 mode/scope setter、不启用计数器或 replay。流程为 runtime 初始化/取得 context → event 属性查询 → 创建禁用的 group → 加入单个 event → 查询 group 原始属性 → finally 销毁。默认 event3，另可独立查询9/10/69；不会把多个事件自动塞到一个组。

供主线程后续串行执行的命令（本文作者未执行 GPU）：`python remote_mcpti_event_preflight.py --maca-root /opt/maca-3.7.1 --event-id 3`。stdout 为 JSON；完整 metadata 成功只代表 `status=metadata_only`，`counter_values=null`、全部 launch/enable 计数为0。查询失败、空指针、异常返回长度和销毁失败分别明确记录，失败退出码为1。字节 buffer 有固定上限，返回状态失败时不把预填内存当作属性值。

离线审核：Ruff 通过；注入假 runtime/PTI 的7项 CPU 测试已保存为 [`test_remote_mcpti_event_preflight.py`](test_remote_mcpti_event_preflight.py)，覆盖 metadata成功但不采集、空context、add失败仍销毁、属性失败不发布缓冲、返回长度越界、销毁失败、success但空group。复算：`python xpuoj_data/bench_records/v727_v729/test_remote_mcpti_event_preflight.py`。测试不加载 MACA 库、不访问GPU；它们验证错误处理，实际 add/query 成功证据来自独立的 event3 preflight JSON，不能混为一谈。

| 阶段 | metric 名称 / metric ID | JSON 中的 event IDs | 本轮要回答的问题 |
| --- | --- | --- | --- |
| A | `achieved_occupancy` / 0 | `3` | 单事件 add/enable/read/disable 是否能成功取得原始值与范围信息 |
| B | `dram_read_bytes` / 8；关联 throughput / 9、transactions / 10 | `9,10` | 两事件能否同采；原始数组是否有效；不重复为三个 metric 采三份事件 |
| C | `stall_sync` / 138 | `69` | 第二个单事件通路能否采集，值是否可复现 |

最终目标去重集合为 `{3,9,10,69}`。A/B/C 是建议的功能验证顺序，**不是已知硬件要求的三个 pass**；是否可合并必须让实际配置结果决定。两 DRAM 事件若不能同组读取，不把不同状态窗口的读数硬拼为一次准确的 bytes/throughput。

第一轮只输出 `raw_event_value` 与状态。event 3 的原始值不是 occupancy 百分比；event 69 的原始值不是 barrier 次数或耗时。全零、特别大的值、缺失值都不是“低占用/无 stall”的证据；应结合返回状态、单位、实例覆盖及重复窗口核验。错误和 unsupported 以 `null + status` 保存，不填 0。

## 4. 只包围一次真实 Stage1 的 smoke 设计

1. 使用独立诊断进程和项目现有 GPU 锁协议；不要与其他采集器并行。保持提交文件本体不变，不向 OJ 文件导入诊断代码。
2. 固定 E32 fixture、seed、routing 和 v720 源码 SHA。输入、权重、输出/workspace、JIT 编译和既有精度检查全部在计数窗口外完成。不能用临时缩小尺寸的测试结果代替真实 E32 Stage1。
3. 从现有 builder/cache 取得真实 Stage1 callable 和原始参数，先执行一次完整预热并等待结束。记录 context/device，确认它们与计数器 group 绑定一致。
4. 第一个 smoke 不需要 callback/replay。先通过第2.3节的 context-scope/mode 证据门槛；未通过则只输出 metadata 并中止，不分配巨量测试输入来等待一个无法安全启用的计数器。通过后 Enable 自动清零；**只启动一次相同 Stage1 callable**；等待它结束；**先 read、后 disable**，不得颠倒。固定1次Stage1预热、1次采集launch；均不运行Stage2。
5. 不用整个 `run_kernel` 当计数窗口，因为它还包含 Stage2。H2D/D2H、参考计算、其他候选和 warmup 不得混入；需要的精度验证在窗口外完成。直接调用 Stage1 仅是分析 harness，不改变 correctness/benchmark 的正式 entry 分支。
6. 对 raw smoke 先不宣称耗时/带宽。若以后需要 duration，必须取得同一个 Stage1 实例的设备区间及确切单位；不能拿整个 entry 时间或旧 trace 的 Stage1 时间当分母。
7. 首个事件通路成功后做短 A/A：同一基线、同样预热/清零流程，至少两个独立窗口。硬件缓存热度、计数器开销或多 pass 均可使窗口不等同于正常 benchmark。再做候选 A/B 时保持初始化与测序一致，带采集的耗时和无采集性能结果分开归档。

最小记录字段：SDK/driver/device/context 身份，切片配置，源码 SHA，kernel 名称、grid/block，fixture/seed/routing，warmup与正式launch序号，metric ID/name，event IDs，domain/instance范围，group/pass序号，各调用返回状态，raw数组/布局，value-kind与单位（未知则 null），是否启用 replay，持续时间及来源（可暂缺）。不记录密码、token、原始输入张量或不必要的内存地址。

## 5. 解释边界与停止条件

- **本集合只用于打通采集和观察变化，不足以区分所有 memory/sync/compute 瓶颈。** 缺少可靠 MMA activity 定义，不能从 occupancy 高或 stall 低推出 Tensor Core 饱和。
- `achieved_occupancy` 不是 profiler trace 的寄存器占用字段；百分比的分母和活跃范围尚未确认。
- `dram_read_bytes/throughput/transactions` 共享事件 9/10，是派生视图而非独立证据。没有单位、转换公式和 kernel 时间，不能公布 GB/s、TB/s、HBM峰值占比。
- `stall_sync` 的名称不能替代 MACA 文档。尤其不能照抄 NVIDIA 同名 metric 的百分比定义，把事件 69 直接解释成 CTA barrier 等待占比。
- 不把 stall 项相加当作总周期分解，也不从 `barrier` 静态条数推动态等待周期。
- 切片仅有25%计算份额不代表 counters 正好覆盖25%整卡资源；不得自动乘4外推。不同 tenant/context 的计数污染若无法排除，要明确标记采集范围不确定。
- 遇到空 context/group、权限不足、unsupported、计数器资源忙、异常 buffer 大小/布局，保存错误后停止该 smoke；不得将失败读数填0，也不连续无界重试。

## 6. 官方资料与 CLI 可行性的边界

官方 [mcProfiler 使用手册](https://developer.metax-tech.com/api/client/document/preview/994/split_files/mcprofiler.html) 说明多指标可导致多轮测量，且对目标程序链接调试库有前置要求。这是 mcProfiler 产品流程，不能不加核验地等同为 MCPTI 手工计数 API 的要求。

官方论坛有 [Linux `mcProfiler perf_exec` 使用记录](https://developer.metax-tech.com/forum/t/c500-mcprofilershi-yong-wen-ti/715/)，其中 `--metrics "Total Cycles"` 后仍出现计数器配置失败。它证明存在此 CLI 形式的使用记录，不证明当前版本支持传入 MCPTI 数字 event ID，也不证明本地容器/切片能成功采集。因此不能把本库存的 `{3,9,10,69}` 直接拼成一个未经帮助文档核对的 CLI 命令。

官方曾 [解释 ISU stall 分类](https://developer.metax-tech.com/forum/t/guan-yu-mcprofilerjie-guo-de-yi-xie-zi-duan-de-yi-wen/674/post/3312/)，但未给出与本库存 event 69 或其他 event IDs 的映射。此文不将两套名称强行对应。
