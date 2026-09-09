# KernelGen 接入已验证，首次优化请求未生成代码

2026-09-09。实际成绩仍为 NSA109/#141658/Accepted86.00，未达到88。

此前“未配置”的判断只覆盖了项目文件，遗漏了已存在的Codex全局连接器。
已纠正：现有连接器通过Windows DPAPI加密凭据启动；认证及tools/list成功，
发现generate_kernel、optimize_kernel、specialize_kernel和autotune_kernel。
未将Token、凭据文件、个人连接路径写入仓库。项目兼容配置也不含明文Token。

当前模型工具清单没有直接暴露这些工具，但已通过同一连接器的MCP SDK
进行一次真实optimize_kernel调用，不是把本地自写代码声称为MCP生成。
调用UTC为2026-09-09T09:44:17.754Z。

## 请求范围

目标为NSA109的nsa_d128_scheduler完整JIT工厂，S1/D128/BS32/G16，
覆盖正式第6点B8/L1024/H1/HQ16。发送了原有工厂及必要TileLang imports，
没有发送凭据或本地私有目录。请求要求保留TileLang而不是转换Triton，
保留参数、causal/索引语义及精度；禁止外部设备代码、异步和结果复用。
附带实际OJ85us/84点和本地约81–85us背景，未虚构分阶段profiling。

这也是兼容性试探：服务的优化工具描述面向Triton，虽接受device=muxi，
尚不能据此断言支持TileLang。没有安装CUDA/Triton环境或改造比赛实现。

## 实际返回

MCP封装`isError=false`，但内部`success=false`，code字段均为null。
服务报告其上游`zyapi.xmsxb.com/v1/messages`返回401 Unauthorized。
因此应判定应用层优化失败，不能只看MCP协议成功。未获得代码或用量数据。

仅尝试一次：该错误没有证据表明重复请求会解决；没有盲目重试、修改模型
凭据或把用户KernelGen Token单独发送到上游域名。服务方需排查上游鉴权。
用户并未要求代其发送第三方故障报告，本任务没有联系服务方。

## 决策

没有生成NSA113提交候选、没有启动相应GPU测试、没有OJ提交。请求编号113
仅用于关联本次服务实验，不是新性能版本。保留109正式基线与112实验文件。
kernelgen-flagos技能要求优化经MCP，故其工作流在服务错误处停止。
若用户要求继续原有TileLang手工流程，应明确该选择，而不是默默转换Triton。
这是一项新的服务端阻碍，不再要求用户重复配置已验证有效的本地Token。

09:46:39UTC重连检查：C500分片仍为25%计算/16000MiB，分片0%且无本任务
测试进程，整卡62%负载。torch2.8.0+metax3.7.1.3、CUDA可用、Triton已安装；
没有因技能前提而更换比赛环境。当前停止不是SSH不可用或需要再跑同一基线。

可公开证据：[kernelgen113_status.json](../results/2026-09-09/kernelgen113_status.json)。
完整请求和返回仅存本机已忽略的`.local`目录；这里的状态文件不含凭据。
