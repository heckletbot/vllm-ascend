# Qwen3.5 模型 PD 分离模式适配方案

本文档给出在 vllm-ascend 上**将 Qwen3.5 系列模型接入 Prefill-Decode（PD）分离部署**的适配方案，包含前置条件、代码侧结论、配置与验证步骤，以及可选增强项。所有路径均为**相对项目根目录（vllm-ascend）的相对路径**。

---

## 一、适配结论概览

| 维度 | 结论 |
|------|------|
| **KV 结构与 Mooncake** | Qwen3.5 使用标准 K/V cache（每层 1 个 K、1 个 V），与 Qwen3 一致，走 Mooncake 的「非 MLA / 非 sparse」分支，**无需**在 connector 里为 Qwen3.5 单独加分支。 |
| **PD 头与 TP 比例** | 使用与 Qwen 相同的 `num_key_value_heads` 与 pd_tp_ratio / pd_head_ratio 逻辑（`ascend_config.py`），**无需**为 Qwen3.5 改 PD 比例计算。 |
| **vllm 版本** | PD 调度与 `kv_transfer_params` 由 vllm 侧下发，需使用**已支持 Qwen3.5 的 vllm 版本**（如 main 或后续发版）；vllm-ascend 使用与之匹配的版本（如 main）。 |
| **vllm-ascend 已做** | 已在单机/非 PD 场景为 Qwen3.5 做 worker、eagle、quant、xlite、eplb 等适配；PD 场景下 **Mooncake 传递路径与现有 Qwen 一致**，无额外模型分支。 |
| **建议** | 先按「标准 Qwen 类模型」做 PD + Mooncake 部署与联调；若实际运行中出现 KV 形状/布局异常，再按第二节做针对性排查与可选增强。 |

---

## 二、前置条件与依赖

### 2.1 vllm 侧

- **Qwen3.5 已合入 vllm**  
  - 至少包含：`Qwen3_5ForConditionalGeneration`、`Qwen3_5MoeForConditionalGeneration` 的 architecture 注册与模型加载。  
  - `model_config` 提供：`hf_text_config.num_hidden_layers`、`num_key_value_heads`、`head_dim` 等；若使用 `get_num_layers()`，需在对应 model config 中实现。  
- **PD 调度与 kv_transfer_params**  
  - 请求能正确带上 `do_remote_prefill` / `do_remote_decode`、`remote_block_ids`、`remote_host`、`remote_port`、`remote_engine_id` 等（与现有 Qwen/DeepSeek PD 行为一致）。  

若 vllm 0.13.0 尚未支持 Qwen3.5，需采用**已支持 Qwen3.5 的 vllm 分支/版本**，并搭配对应版本的 vllm-ascend（如 main）。

### 2.2 vllm-ascend 侧（相对路径）

- **已完成的 Qwen3.5 相关改动**（单机/非 PD 已用）：  
  - `vllm_ascend/worker/worker.py`：Qwen3.5 架构的 KV cache 内存折半。  
  - `vllm_ascend/spec_decode/eagle_proposer.py`：Qwen3.5 MoE 的 image_token 支持。  
  - `vllm_ascend/quantization/quant_config.py`：qwen3_5_moe 量化配置。  
  - `vllm_ascend/xlite/xlite.py`：Qwen3.5 架构的 xlite strategy_map。  
  - `vllm_ascend/eplb/utils.py`：qwen3_5_moe 的 EPLB 支持。  
- **PD + Mooncake 相关代码**（无需为 Qwen3.5 新增模型分支）：  
  - `vllm_ascend/distributed/mooncake_connector.py`  
  - `vllm_ascend/distributed/mooncake_transfer_engine.py`  
  - `vllm_ascend/distributed/mooncake_layerwise_connector.py`（若使用逐层传递）  
  - `vllm_ascend/ascend_config.py`（pd_tp_ratio / pd_head_ratio，已支持「非 MLA」的 Qwen 类）  

### 2.3 运行环境

- Mooncake 与 Ascend Direct Xfer 可用；当前 **Mooncake 不支持 IPv6**，P/D 节点需使用 IPv4。  
- CANN、torch-npu、vllm、vllm-ascend 版本与官方文档/Release 一致。

---

## 三、为何无需在 Mooncake Connector 中加 Qwen3.5 分支

### 3.1 KV cache 结构推断（相对路径）

**文件**：`vllm_ascend/distributed/mooncake_connector.py`  

- **register_kv_caches**（约 1201–1255 行）根据 `kv_caches` 的**结构**自动判断：  
  - `use_mla`：每层 2 个 cache 且最后一维不同（如 MLA）。  
  - `use_sparse`：每层 3 个 cache（稀疏布局）。  
  - **否则**：标准布局，`block_len = [kv_elem_size * prod(block_shape)]`，即单元素。  
- Qwen3.5 与 Qwen3 相同，为**每层标准 K/V 两个 tensor**，走 **else** 分支，`use_mla=False`、`use_sparse=False`，**无需**按模型名加分支。

### 3.2 tp_num_need_pulls 与头划分（相对路径）

**文件**：`vllm_ascend/distributed/mooncake_connector.py`  

- **tp_num_need_pulls**（约 1161–1169 行）：仅对 `is_deepseek_mla` 特殊为 1；否则按 `num_key_value_heads` 与 P/D 的 tp_size 计算。  
- Qwen3.5 非 MLA，使用 `hf_text_config.num_key_value_heads` 即可，与现有 Qwen 一致。  

**文件**：`vllm_ascend/ascend_config.py`  

- **pd_tp_ratio / pd_head_ratio**（约 96–125 行）：在 `not model_config.is_deepseek_mla` 时计算，并依赖 `hf_text_config.num_key_value_heads`；注释已说明当前「only support Qwen model」，Qwen3.5 属于该逻辑覆盖范围。

### 3.3 其他模型相关点

- **enable_kv_nz**：仅支持 `is_deepseek_mla`，Qwen3.5 不启用则无影响。  
- **get_num_layers**：若使用 layerwise connector，依赖 `model_config.get_num_layers()`；vllm 侧 Qwen3.5 的 model config 需实现该接口（通常与 `num_hidden_layers` 一致）。  

综上，**无需在 Mooncake 或 ascend_config 中为 Qwen3.5 增加单独分支**，按「标准 Qwen 类模型」即可。

---

## 四、推荐适配步骤（按顺序执行）

### 步骤 1：确认 vllm 与 vllm-ascend 版本

- 使用**已支持 Qwen3.5** 的 vllm（如 main 或后续发版）。  
- 使用与之匹配的 vllm-ascend（如 main），并确认已包含上述 Qwen3.5 相关单机改动（worker、eagle、quant、xlite、eplb）。  

### 步骤 2：单机验证 Qwen3.5

- 在单机、非 PD 场景下跑通 Qwen3.5（含 MoE 若需要）。  
- 确认无 OOM、无错误，再进入 PD 联调。

### 步骤 3：PD 配置（与现有 Qwen 保持一致）

- **kv_transfer_config**：  
  - `connector` 使用 `MooncakeConnectorV1`（或逐层场景下 `MooncakeLayerwiseConnector`）。  
  - `kv_port`、`engine_id`、`kv_role`（`kv_producer` / `kv_consumer`）正确。  
- **extra_config** 中 prefill / decode 的 **tp_size、dp_size、pp_size** 与当前部署一致；decode 的 **pp_size 必须为 1**。  
- 参考现有 PD 文档与示例：  
  - `docs/source/user_guide/feature_guide/pd_mooncake_cache_transfer.md`  
  - `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md`  

### 步骤 4：P 节点与 D 节点启动

- P 节点：`kv_role=kv_producer`，加载 Qwen3.5 模型，配置 prefill 的 tp/dp/pp。  
- D 节点：`kv_role=kv_consumer`，同样加载 Qwen3.5，配置 decode 的 tp/dp。  
- 两边 **kv_transfer_config**、**extra_config** 中 prefill/decode 的规模需一致且符合约束（如 prefill_tp_size % decode_tp_size == 0）。  

### 步骤 5：联调与验证

- 使用短序列、固定 seed 做 P→D 的端到端推理，对比单机结果。  
- 检查日志中是否有 Mooncake 报错、超时或 block 映射错误；若有，再按下一节做可选增强与排查。

---

## 五、可选增强与异常排查

### 5.1 若 Qwen3.5 使用特殊 KV 布局

若未来 Qwen3.5 采用「每层多块」或与 MLA/sparse 类似的布局，且 vllm 暴露的 `kv_caches` 结构发生变化：

- **位置**：`vllm_ascend/distributed/mooncake_connector.py` 的 `register_kv_caches`。  
- **做法**：在现有 `use_mla` / `use_sparse` 判断后，增加对 Qwen3.5 的**结构分支**（例如根据 `len(first_kv_cache_tuple)` 或 `model_config.architecture` 推断 block_len / 每层 cache 数），并保证 `block_len`、`kv_caches_base_addr` 与真实布局一致。  
- **同步**：若使用 layerwise，需在 `vllm_ascend/distributed/mooncake_layerwise_connector.py` 中做相同结构推断。

### 5.2 若 P/D 头数或 TP 比例异常

若实测发现 Qwen3.5 在 PD 下出现头划分错误或 tp_num_need_pulls 异常：

- **位置**：  
  - `vllm_ascend/distributed/mooncake_connector.py` 中 `tp_num_need_pulls` 与 `_get_remote_tp_ranks` 等。  
  - `vllm_ascend/ascend_config.py` 中 `pd_head_ratio` / `num_head_replica`。  
- **做法**：在**确有需要**时，为 Qwen3.5 增加与 `is_deepseek_mla` 类似的 `is_qwen3_5_xxx` 分支（仅当 Qwen3.5 的头划分与标准 Qwen 不同时使用）。  

### 5.3 LinearAttention 与双倍 KV 内存

- 当前在 **worker** 中已对 Qwen3.5 架构做 `available_kv_cache_memory // 2`，仅影响**容量规划**，不改变 KV 的**逻辑块布局**。  
- 若 D 节点在接收后做 reformat（如 concat、NZ），仍走现有 `reformat_kv_cache` 分支；仅当 Qwen3.5 引入新的 cache 格式时，才需在 `vllm_ascend/distributed/mooncake_connector.py` 的 `reformat_kv_cache` / `_cat_kv_cache` / `_nz_kv_cache` 中扩展。

### 5.4 文档与配置示例

- 在 **PD 部署文档** 或 **Qwen 教程** 中增加「Qwen3.5 适用」说明及推荐 P/D TP 比例、限制（如 IPv6、decode pp_size=1）。  
- 在 **examples** 下可增加 Qwen3.5 的 PD 启动示例（若与 Qwen3 仅模型名不同，可复用现有示例并注明模型名与版本）。  

---

## 六、涉及文件相对路径汇总

| 用途 | 相对路径 |
|------|----------|
| Worker 侧 Qwen3.5 KV 内存折半 | `vllm_ascend/worker/worker.py` |
| Eagle proposer Qwen3.5 MoE | `vllm_ascend/spec_decode/eagle_proposer.py` |
| 量化 qwen3_5_moe | `vllm_ascend/quantization/quant_config.py` |
| xlite Qwen3.5 架构映射 | `vllm_ascend/xlite/xlite.py` |
| EPLB qwen3_5_moe | `vllm_ascend/eplb/utils.py` |
| Mooncake 非逐层 Connector | `vllm_ascend/distributed/mooncake_connector.py` |
| Mooncake 逐层 Connector | `vllm_ascend/distributed/mooncake_layerwise_connector.py` |
| Mooncake 传输引擎 | `vllm_ascend/distributed/mooncake_transfer_engine.py` |
| PD 头/TP 比例（含 Qwen） | `vllm_ascend/ascend_config.py` |
| PD Mooncake 传递说明 | `docs/source/user_guide/feature_guide/pd_mooncake_cache_transfer.md` |
| PD 部署示例 | `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md` |

---

## 七、总结

- **默认结论**：Qwen3.5 在 PD 分离模式下**按标准 Qwen 类模型**使用现有 Mooncake 与 PD 逻辑即可，**无需**在 connector 或 ascend_config 中为 Qwen3.5 新增模型分支。  
- **前提**：vllm 已支持 Qwen3.5（含 PD 的 kv_transfer_params），vllm-ascend 已合入前述 Qwen3.5 单机改动。  
- **建议**：先按本文「推荐适配步骤」完成部署与联调；仅在实际出现 KV 结构或头划分异常时，再按「可选增强与异常排查」做最小化修改，并同步更新文档与示例。

---

*文档基于当前 vllm-ascend 与 Mooncake 逻辑整理；若 vllm 或 vllm-ascend 后续对 Qwen3.5 的 KV 布局或 PD 接口有变更，需按实际代码再复核。*
