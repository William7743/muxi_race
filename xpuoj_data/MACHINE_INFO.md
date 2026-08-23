# 评测机器硬件档案（2026-08-23 记录）

来源：XPU-OJ 机器列表页 + 评测机 `/proc/cpuinfo` 实测输出。

## 机器列表（4 台，全部在线）

| 机器 | CPU | 内存 | 系统 | 状态 |
|---|---|---|---|---|
| C500 (1) | Intel Xeon Gold 6530 @ 64x 4GHz | 2063870 MiB (~2 TB) | Linux 5.15.0-58-generic | 在线 |
| C500 (2) | Intel Xeon Gold 6530 @ 64x 4GHz | 2063870 MiB (~2 TB) | Linux 5.15.0-58-generic | 在线 |
| C500 (3) | Intel Xeon Gold 6530 @ 64x 4GHz | 2063870 MiB (~2 TB) | Linux 5.15.0-58-generic | 在线 |
| C500 (4) | Intel Xeon Gold 6530 @ 64x 4GHz | 2063870 MiB (~2 TB) | Linux 5.15.0-58-generic | 在线 |

- GPU：沐曦 C500（每机 3 卡，状态列显示 3/3）
- 4 台机器配置完全一致 → 评测可能调度到不同机器，是「同代码多次提交结果漂移」的候选解释之一

## 宿主 CPU 缓存（/proc/cpuinfo 汇总）

```
L1d    3072 KiB   (48 KB/core × 64)
L1i    2048 KiB   (32 KB/core × 64)
L2    131072 KiB  (2 MB/core × 64)
L3    327680 KiB  (320 MB shared, Emerald Rapids)
```

## CPU flags（完整）

```
fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
```

## 对优化工作的启示

1. **宿主 CPU 是 Emerald Rapids（第五代至强）**：64 线程 @ 4GHz、320MB L3、~2TB 内存。
   评测的 host 侧数据准备（路由表、stacked tokens 组装、校验）在强 CPU 上开销小，
   进一步印证瓶颈在 GPU kernel 本身，Python 侧优化空间有限（与 v185 WA 结论一致）。
2. **多机调度**：4 台同配机器，提交可能落到不同机器。机器间细微差异（温度/
   占用/驱动状态）叠加 TileLang-MACA lowering 非确定性，是复验结果漂移的
   候选解释。后续「两次独立 Accepted 才提升稳定版」的纪律继续有效。
3. 本档案为 host 侧信息；GPU（C500）侧已知档案见 `OPTIMIZATION_LOG.md`
   硬件档案节（104 SM、warp_size=64、128K regs/SM、64KB smem/block、
   DRAM ~1.4-1.8TB/s）。
