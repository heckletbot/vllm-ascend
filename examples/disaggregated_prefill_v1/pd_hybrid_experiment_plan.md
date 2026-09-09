# PD 与混部自适应路由实验方案

## 1. 实验目标

本实验用于判断以下两类混部路由是否优于纯 PD 分离：

1. P 节点空闲时，将长输入、短输出请求直接交给 P 本地完成，避免 KV Cache 传输和 D 排队。
2. 指定 D 实例已经命中 Prefix Cache 时，将请求直接交给该 D 本地完成，跳过 P 和 KV Cache 传输。

实验 Agent 不应预设混部一定更好。每个场景必须先确认指标有效、目标负载状态成立，再进行性能比较。
若场景没有构造出预期状态，Agent 应调整流量参数并重新验证，而不是继续使用无效数据得出结论。

本文中的“Prefix Cache 命中”是指 vLLM Prefix Cache 命中，不是 Perplexity 指标。

## 2. 环境与固定条件

- 模型：Qwen3.6-35B-A3B。
- 硬件：单台 A3，共 16 张卡。
- P：DP2 × TP4 × EP8，共 8 张卡，对外提供 2 个 DP 端口。
- D：DP4 × TP2 × EP8，共 8 张卡，对外提供 4 个 DP 端口。
- P/D 使用固定且可记录的模型、显存、batch、Prefix Cache、异步调度和 Expert Parallel 配置。
- P/D 均具备处理无 `kv_transfer_params` 本地请求的能力。
- PD 请求通过 `MooncakeConnectorV1` 传输 KV Cache。
- Local-P 和 Local-D 请求不携带 `kv_transfer_params`。
- 使用自生成固定数据集和自定义发送脚本，不使用 AIPerf。

每轮实验前必须保存完整启动命令、环境变量、模型版本、代码 commit、设备映射和服务日志。
除当前正在验证的变量外，不允许在对照组之间改变 Worker 配置。

## 3. 路由定义

| 路由 | 请求路径 | KV 传输 |
| --- | --- | --- |
| `pd` | RS → P → D | 是 |
| `local_p` | RS → P，本地 Prefill + Decode | 否 |
| `local_d` | RS → D，本地 Prefill + Decode | 否 |

对同一批请求进行 A/B 对比时，必须复用相同 prompt、请求顺序、并发、输出长度和随机种子。
模式切换前应重启服务或可靠地清理 Prefix Cache，避免前一轮缓存污染后一轮。

## 4. 请求数据

### 4.1 冷 Prefix 数据

- 输入长度：8K、16K、32K、64K、128K token。
- 输出长度：固定 1K token。
- 并发：4、8、16、32。
- 每个输入长度至少准备 320 条独立 prompt。
- 每条 prompt 应在第一个 Prefix Cache block 内不同。
- 套用与服务一致的 Chat Template 后重新分词，确认实际输入 token 数。
- 设置 `temperature=0`、固定 seed 和 `ignore_eos=true`。

### 4.2 Prefix Cache 数据

每个长度生成一组：

- 共享前缀：32K、64K、128K token。
- 独立 suffix：约 1K token。
- 输出长度：1K token。
- 同一组请求共享完全相同的 token 前缀，但 suffix 和 request ID 不同。

Prefix hash 依赖实际 token 序列。模型、Tokenizer、Chat Template、LoRA 和目标 DP endpoint 必须保持一致。

## 5. 阶段 0：指标有效性验证

性能实验开始前，Agent 必须先确认观测指标能反映真实状态。

### 5.1 客户端指标

发送脚本逐请求记录：

- 请求 ID、路由和目标 DP endpoint。
- 实际输入/输出 token 数。
- 请求发送时间、首 token 时间和完成时间。
- TTFT。
- TPOT：`(E2E - TTFT) / (output_tokens - 1)`。
- E2E。
- HTTP 状态、流是否正常结束和错误信息。

### 5.2 vLLM 指标

每秒采集每个 P/D DP endpoint 的 `/metrics`：

```bash
curl -s http://HOST:PORT/metrics | \
grep -E 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|prefix_cache_queries|prefix_cache_hits)'
```

重点指标：

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:num_requests_waiting_by_reason`
- `vllm:kv_cache_usage_perc`
- `vllm:prefix_cache_queries`
- `vllm:prefix_cache_hits`

多 DP 场景必须按具体 endpoint/engine 观测，不能只看聚合平均值。

### 5.3 NPU 与连接器指标

使用真机已有的低开销 NPU 监控每秒采集：

- 每张卡的计算利用率。
- HBM 使用量。
- 通信利用率。

同时从 Mooncake 和 Scheduler 日志检查：

- KV 发送/接收请求数。
- delay-free block 数及持有时间。
- 请求完成后 KV Cache 是否回落。
- ACL Graph replay、OOM、preemption 和 recompute 错误。

### 5.4 指标验证步骤

1. 服务完全空载时采集基线，确认 waiting/running 为 0，NPU 利用率接近空载水平。
2. 串行发送一个短请求，确认 running、token 计数和客户端时延发生合理变化。
3. 提高并发制造排队，确认 waiting 能够升高并在流量结束后回落。
4. 向同一 endpoint 连续发送相同前缀，确认 `prefix_cache_queries` 和 `prefix_cache_hits` 增量合理。
5. 向不同 DP endpoint 发送相同前缀，确认不会错误地将跨 endpoint 请求判定为同一缓存命中。

如果指标缺失、长期不更新或与请求行为矛盾，Agent 必须先修复采集方式或改用日志佐证。
指标未验证前不得进入后续性能实验。

## 6. 阶段 1：混合请求稳定性

按 `pd : local_p : local_d = 1 : 1 : 1` 同时发送三类请求。

建议初始组合：

- 并发：8、16。
- 输入：8K、32K、128K。
- 输出：1K。

通过 request ID 的稳定 hash 决定路由，保证重复实验的比例和顺序一致。

通过条件：

- P/D 服务不崩溃。
- 所有请求完成，无永久 pending。
- 流式响应正常结束。
- Local-P/Local-D 不进入 Mooncake send/recv。
- PD 请求正常传输并最终释放 KV。
- 实验结束后 waiting、running 和 KV Cache 使用量回落。

任一条件失败时，Agent 应保留请求 ID、服务日志和指标时间窗，停止性能比较并定位问题。

## 7. 阶段 2：构造并验证 P 空闲场景

### 7.1 初始场景

使用短输入、长输出的 PD 背景流量，使 P 很快完成 Prefill，而 D 长时间执行 Decode：

- 背景输入：1K。
- 背景输出：4K。
- 背景并发：从 4 开始，依次尝试 8、16。
- 背景路由：全部 `pd`。

注入用于比较的请求：

- 输入：32K、64K、128K。
- 输出：1K。
- 注入并发：1、2、4。
- 对比路由：`pd` 与 `local_p`。

### 7.2 P 空闲判定

以下阈值仅作为初始参考，Agent 应根据空载和满载基线调整：

- P 的 `num_requests_waiting` 连续一个观测窗口为 0。
- P 的 NPU 利用率明显低于其满载基线。
- P 的 running 和 KV Cache 使用率没有持续增长。
- 同期 D 的 running、利用率或 waiting 明显高于 P。

“P 空闲”不能仅根据单个瞬时 NPU 利用率判断。至少结合 waiting、running、NPU 利用率和 KV Cache。

### 7.3 自适应调整规则

如果 P 与 D 都空闲：

- 增加背景并发，或增加背景输出长度。

如果 P 也持续繁忙：

- 减少背景输入长度。
- 降低背景并发。
- 确认 P 没有被 Local-P 探测请求长期占用。

如果 D 已经 OOM、持续 preemption 或大量超时：

- 降低背景并发或输出长度。
- 不得把异常饱和状态当作有效“D 忙”场景。

如果多次调整仍无法形成“P 有余量、D 稳定繁忙”的状态，应记录该拓扑下场景不可稳定构造，并停止该项结论。

### 7.4 希望回答的问题

- 在 P 有余量时，Local-P 是否降低长输入、短输出请求的 TTFT/E2E？
- Local-P 是否减少 Mooncake KV 传输量？
- Local-P 是否导致背景 PD 请求 TTFT 或 P waiting 明显恶化？
- 随注入并发增加，Local-P 的收益在哪个点消失？

只有 Local-P 自身性能改善且没有明显破坏背景流量 SLO，才能认为该场景适合路由到 P。

## 8. 阶段 3：构造并验证 D Prefix Cache 命中场景

### 8.1 缓存预热

1. 选择一个固定 D DP endpoint。
2. 在低负载下向该 endpoint 发送共享前缀请求进行预热。
3. 不重启服务、不切换 endpoint。
4. 立即发送“相同共享前缀 + 独立 suffix”的探测请求。

探测组合：

- 共享前缀：32K、64K、128K。
- suffix：约 1K。
- 输出：1K。
- 并发：先 1，再尝试 4、8、16。
- 对比路由：`pd` 与固定 endpoint 的 `local_d`。

### 8.2 Prefix Cache 有效命中判定

按实验前后的 counter 增量计算：

```text
hit_rate = Δprefix_cache_hits / Δprefix_cache_queries
```

必须同时确认：

- 请求被发到预热时相同的 D endpoint。
- 实际 token 前缀完全相同。
- Prefix Cache 已启用。
- `prefix_cache_hits` 有明显增量。
- 服务日志中的 cached token 数与共享前缀长度相符。

不能仅因为请求内容“看起来相似”就认定命中。

### 8.3 命中无效时的自适应调整

如果没有有效命中，Agent 应依次检查：

1. 是否错误路由到了其他 DP endpoint。
2. Chat Template、系统消息或 Tokenizer 是否导致 token 前缀不同。
3. 共享前缀是否至少覆盖完整 cache block。
4. 预热与探测之间是否有高负载导致缓存淘汰。
5. KV Cache 使用率是否过高。
6. 服务是否发生重启、preemption 或缓存重置。

调整方式：

- 先用完全相同请求串行验证最基本命中。
- 降低并发和 KV 压力。
- 增大共享前缀并缩短预热到探测的间隔。
- 使用固定 endpoint，禁用 RS 的二次负载均衡。

如果仍无法证明有效命中，不得使用该轮数据比较 D-Local 与 PD；应输出诊断结果和下一步建议。

### 8.4 希望回答的问题

- D Prefix Cache 有效命中后，Local-D 是否显著降低 TTFT？
- Local-D 是否省去 P 排队和 Mooncake 传输？
- Local-D Prefill/suffix 计算是否影响已有 Decode 请求的 TPOT？
- 随 D 并发和 KV 使用率增加，缓存命中收益何时消失？

## 9. 结果判断

每个有效组合至少重复 3 轮。Agent 应同时报告原始值、P50、P99 和三轮波动，不能只报告平均值。

路由被判定为更优时，应同时满足：

1. 目标请求 TTFT 或 E2E 有稳定改善。
2. 改善超过实验噪声，三轮方向一致。
3. 背景请求的 TTFT/TPOT 不违反既定 SLO。
4. 服务无错误、OOM、异常 preemption 或 KV 泄漏。
5. 目标前提已经成立，例如 P 确实空闲或 D Prefix Cache 确实命中。

若不同指标冲突，优先满足线上 SLO，再比较吞吐和资源利用率。

实验最终应形成如下路由结论，而不是强制证明某条路径更好：

```text
(输入长度, 输出长度, P负载, D负载, Prefix位置, KV使用率)
    → pd / local_p / local_d
```

## 10. Agent 输出物

实验 Agent 应保存：

- `requests.jsonl`：固定输入数据。
- `request_results.jsonl`：逐请求路由和时延。
- `metrics/`：每个 DP endpoint 的 `/metrics` 时间序列。
- `npu_metrics/`：逐卡利用率和内存时间序列。
- P、D、RS 和 Mooncake 日志。
- 每轮完整启动命令和配置。
- `summary.md`：有效场景、无效场景、调整过程、结果表和建议路由边界。

`summary.md` 必须明确区分：

- 已由数据证明的结论。
- 因场景构造失败而无法判断的项目。
- 尚未验证的假设。
- 建议继续调整的输入、并发或负载条件。
