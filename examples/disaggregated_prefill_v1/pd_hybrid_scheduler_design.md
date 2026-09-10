# PD 与混部负载感知调度初期设计

## 1. 背景

当前 RS 默认将请求按 PD 分离路径发送：

```text
RS → P Prefill → Mooncake KV 传输 → D Decode
```

实验表明，当 D 正在处理大量 Decode、而 P 已经空闲时，将部分长输入、短输出请求直接交给 P
本地完成，可以减少 D 排队和 KV 传输，并改善请求 E2E。

本设计在现有 PD 调度算法前增加一个可关闭的负载感知决策层：

```text
                         ┌→ Local-P
请求 → 混部决策（可关闭）├→ Local-D
                         └→ 原有 PD 调度
```

新算法只决定请求路径，不改变 P/D Worker 角色，不动态切换 `kv_role`。

## 2. 设计目标

- 通过总开关控制新算法，默认关闭。
- 关闭时完整保留现有 PD 行为。
- P 空闲时允许合适的请求直接在 P 完成。
- D 空闲时允许合适的请求直接在 D 完成。
- 无法可靠判断负载时回退现有 PD 调度。
- 新逻辑以插入方式实现，避免重写现有 P/D 选择、重试和响应转发逻辑。
- 防止并发请求同时看到“空闲”并过量发送到同一个节点。
- 提供完整决策日志和指标，支持后续调整阈值。

## 3. 非目标

- 不修改 Mooncake Connector。
- 不修改 P/D KV send/recv 线程。
- 不修改 D 的远端 KV 等待流程。
- 第一版不基于 Prefix Cache 做路由。当前 Qwen3.6 hybrid Mamba 模型在 vLLM v0.23.0
  的 `mamba_cache_mode=align` 下未观测到有效 Prefix Cache 命中。
- 第一版不追求全局最优，只提供保守、可回退的利用率优化。

## 4. 配置开关

建议使用 RS 命令行或配置文件参数，不新增 vLLM Worker 环境变量。

```yaml
pd_hybrid_routing:
  enabled: false
  enable_local_p: true
  enable_local_d: false
  local_d_canary_ratio: 0.0

  metrics_poll_interval_seconds: 1.0
  metrics_stale_seconds: 3.0
  idle_consecutive_samples: 3

  idle_max_waiting: 0
  idle_max_running_p: 0
  idle_max_running_d: 0
  idle_max_kv_usage: 0.50

  local_p_max_inflight_per_endpoint: 2
  local_d_max_inflight_per_endpoint: 1

  local_p_min_input_tokens: 32768
  local_p_max_output_tokens: 1024
  local_d_max_input_tokens: 32768
  local_d_max_output_tokens: 1024
```

总开关：

```text
enabled=false → 所有请求使用原有 PD 算法
enabled=true  → 先尝试混部决策，未命中时回退原有 PD 算法
```

`enable_local_p` 和 `enable_local_d` 应相互独立。现有实验已经证明 Local-P 在特定场景下有
E2E 收益；Local-D 目前只有功能稳定性证据，因此建议默认关闭或仅以小比例 canary 开启。

## 5. 插入位置

新决策应放在原有 `assign_instances` 或等价 PD 分配函数之前：

```python
async def route_request(request):
    if not hybrid_router.enabled:
        return await existing_pd_route(request)

    plan = hybrid_router.try_reserve(request)
    if plan.route == "local_p":
        return await local_route(request, plan.endpoint)
    if plan.route == "local_d":
        return await local_route(request, plan.endpoint)

    return await existing_pd_route(request)
```

原有 PD 路径保持不变。Local-P/Local-D 直接发送客户端原始 body，不注入
`kv_transfer_params`，并保留原始 `max_tokens`、`stream` 和 `stream_options`。

以下请求不得再次参与混部决策：

- 已携带 `kv_transfer_params` 的内部 PD 请求。
- 重算、释放 KV 或其他内部控制请求。
- 已经开始向客户端返回流式数据的重试请求。

## 6. 负载数据

### 6.1 后台采集

RS 后台定时采集每个 DP endpoint 的：

- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:num_requests_waiting_by_reason`
- `vllm:kv_cache_usage_perc`
- 指标采集时间和 endpoint 健康状态

NPU 利用率可用于观测和调参，但第一版不作为硬路由条件。瞬时 NPU 利用率存在延迟和波动，
同步查询也会增加请求路径开销。

### 6.2 RS 本地计数

RS 同时维护比 `/metrics` 更新更及时的本地状态：

- 已分配但后端指标尚未反映的 Local-P 请求数。
- 已分配但后端指标尚未反映的 Local-D 请求数。
- 每个 endpoint 的 RS in-flight 请求数。
- 请求预计剩余输出 token 数。

后续可增加：

- P 待处理 Prefill token 数。
- P 本地剩余 Decode token 数。
- D 剩余 Decode token 数。

仅使用 `running` 请求数无法区分一个短请求和一个 128K/1K 请求，因此 token backlog
应作为第二版负载模型。

## 7. 空闲状态定义

### 7.1 Endpoint 状态机

每个 endpoint 有三种状态：

```text
IDLE / BUSY / UNKNOWN
```

`UNKNOWN`：

- endpoint 不健康；或
- 指标年龄超过 `metrics_stale_seconds`；或
- 指标解析失败。

`UNKNOWN` endpoint 不接收 Local 请求，但仍由原有健康检查和 PD 调度决定是否可用。

初期 `IDLE` 条件：

```text
waiting <= idle_max_waiting
running <= 对应角色的 idle_max_running
kv_cache_usage <= idle_max_kv_usage
local_reserved < local_max_inflight_per_endpoint
连续 idle_consecutive_samples 次满足
```

任一情况应立即进入 `BUSY`：

- `waiting > 0`
- KV Cache 超过水位
- running 超过阈值
- 本地预留达到上限

连续采样进入 IDLE、立即退出 IDLE，可以避免节点在边界附近频繁抖动。

### 7.2 P 空闲

P 集群存在至少一个 `IDLE` 的 P endpoint，即认为存在可用 P 空闲容量。

第一版保守要求目标 P endpoint：

```text
waiting = 0
running = 0
kv_cache_usage < 50%
local_p_reserved < 2
```

后续根据实验逐步放宽 `running`，不应一开始就使用 NPU 利用率百分比代替队列状态。

### 7.3 D 空闲

D-Local 会在 D 上引入 Prefill，可能干扰正在运行的 Decode，因此第一版要求目标 D endpoint
真正空闲：

```text
waiting = 0
running = 0
kv_cache_usage < 50%
local_d_reserved < 1
```

在验证 D-Local 对背景 TPOT 的影响前，不应将“D waiting=0 但仍有大量 Decode running”
定义为 D 空闲。

## 8. 请求准入

### 8.1 Local-P

第一版只接收已有实验支持的请求：

```text
input_tokens >= 32K
requested max_tokens <= 1K
存在 IDLE P endpoint
```

Local-P 适用于 D Decode 繁忙、P 已完成当前 Prefill 工作的场景。短输出限制用于控制请求在 P
上持续 Decode 的时间，避免侵占后续 PD Prefill。

### 8.2 Local-D

第一版采用更保守限制：

```text
input_tokens <= 32K
requested max_tokens <= 1K
存在完全 IDLE D endpoint
通过 Local-D canary 比例
```

在获得有效收益和背景 TPOT 数据前，不扩大输入长度、输出长度和并发。

### 8.3 不适合本地执行

以下情况直接回退 PD：

- 输出长度未知或超过上限。
- 没有新鲜、可信的后端指标。
- P/D endpoint 均繁忙。
- KV Cache 使用率超过水位。
- 本地预留已经达到上限。
- 请求类型或模型能力不支持本地完整推理。

## 9. 初期决策算法

```python
def decide(request, snapshot):
    if not config.enabled:
        return PD("feature_disabled")

    if request.kv_transfer_params is not None:
        return PD("internal_pd_request")

    idle_p = select_least_loaded_idle_p(snapshot)
    idle_d = select_least_loaded_idle_d(snapshot)

    p_eligible = (
        config.enable_local_p
        and idle_p is not None
        and request.input_tokens >= config.local_p_min_input_tokens
        and request.max_tokens <= config.local_p_max_output_tokens
    )
    d_eligible = (
        config.enable_local_d
        and idle_d is not None
        and request.input_tokens <= config.local_d_max_input_tokens
        and request.max_tokens <= config.local_d_max_output_tokens
        and pass_canary(request, config.local_d_canary_ratio)
    )

    if p_eligible:
        return reserve_atomically(LOCAL_P, idle_p)
    if d_eligible:
        return reserve_atomically(LOCAL_D, idle_d)
    return PD("no_eligible_idle_backend")
```

第一版优先 Local-P，因为它已有实验收益，且 D-Local Prefill 对 Decode 的影响尚未量化。
如果后续证明 D-Local 有收益，可在 P/D 同时空闲时按照预测完成时间选择，而不是固定优先级。

## 10. 原子预留

负载快照与路由决策之间存在时间差。多个并发请求可能同时看到同一个 endpoint 为 IDLE。

`try_reserve` 必须在同一个锁或原子操作中完成：

1. 重新检查 endpoint 状态。
2. 检查本地预留上限。
3. 增加 `local_reserved`。
4. 返回路由计划。

请求完成、失败或取消时必须在 `finally` 中释放预留。若预留泄漏，节点会永久失去 Local
流量；因此需要预留超时和周期性校正机制。

## 11. 失败与回退

- 决策阶段异常：回退原有 PD。
- Local 请求发送前连接失败：释放预留，可回退 PD。
- Local 请求收到首个响应字节前返回明确 5xx：是否回退由配置控制，并记录一次失败。
- 流式响应已经开始：不得自动切换到 PD，避免重复输出。
- endpoint 指标过期：停止新的 Local 准入。
- 连续 Local 错误超过阈值：自动关闭对应 `enable_local_p` 或 `enable_local_d`，保留 Pure-PD。

实验模式可以禁用自动失败回退，避免掩盖 Local 路径错误；生产模式再启用首包前回退。

## 12. 可观测性

每个请求记录结构化日志：

```json
{
  "request_id": "req-001",
  "route": "local_p",
  "decision_reason": "p_idle_long_input_short_output",
  "input_tokens": 65536,
  "max_tokens": 1024,
  "endpoint": "p-dp0",
  "running": 0,
  "waiting": 0,
  "kv_cache_usage": 0.12,
  "local_reserved": 1,
  "metrics_age_ms": 380
}
```

建议增加 RS 指标：

- `rs_route_requests_total{route,reason}`
- `rs_local_inflight{role,endpoint}`
- `rs_backend_state{role,endpoint,state}`
- `rs_route_fallback_total{reason}`
- `rs_route_decision_latency_seconds`

性能结果必须按 `pd`、`local_p`、`local_d` 分开统计 TTFT、TPOT、E2E、成功率和取消率。

## 13. 实施步骤

### 第一步：旁路观测

- 增加配置和后台负载采集。
- 算法只输出“如果开启会选择什么”的 shadow 日志。
- 所有真实请求仍走原有 PD。
- 核对 IDLE/BUSY/UNKNOWN 是否与服务实际状态一致。

### 第二步：Local-P canary

- 开启总开关和 Local-P。
- `local_p_max_inflight_per_endpoint=1`。
- 仅接收 32K～128K 输入、最大 1K 输出。
- 验证背景 PD 请求 SLO和 Local-P E2E。
- 逐步将每 endpoint 上限增加到 2、4，寻找边界。

### 第三步：Local-D canary

- Local-P 稳定后再单独开启 Local-D。
- 从完全空闲 D、输入不超过 32K、并发1开始。
- 重点观测已有 Decode 请求的 TPOT。
- 无稳定收益时保持 Local-D 关闭。

### 第四步：负载模型升级

- 从请求数阈值升级为 token backlog。
- 估计 Local-P 与 PD 的完成时间。
- 增加进入/退出混部的滞回阈值。
- 根据实验结果调整输入、输出、KV水位和并发上限。

## 14. 验收标准

功能开关关闭时：

- 路由和现有版本一致。
- 不增加 Local 请求。
- 不改变请求 body 和响应。

功能开关开启时：

- Local-P/Local-D 不携带 `kv_transfer_params`。
- Local 请求不触发 Mooncake传输。
- 并发预留不超过配置上限。
- 指标失效或后端繁忙时稳定回退 PD。
- 服务无崩溃、永久 pending、KV泄漏或重复流式输出。

性能方面希望得到：

- P 空闲、D 繁忙时，Local-P 稳定改善目标请求 E2E。
- Local-P 不显著破坏背景 PD 请求的 TTFT/TPOT SLO。
- 节点从空闲转为繁忙后，新请求能够及时回退 PD。
- Local-D 只有在实验证明有收益后才扩大启用范围。

## 15. 待实验确定的参数

- P/D running 空闲阈值。
- P/D KV Cache水位。
- 每个 endpoint 的 Local in-flight 上限。
- Local-P 最大输出长度。
- Local-D 最大输入和输出长度。
- 指标采样窗口和过期时间。
- Local-D canary比例。
- P/D 同时空闲时的选择规则。

这些参数不得仅凭经验固定，应由后续实验数据确定，并保留配置化能力。
