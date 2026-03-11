# vllm-ascend-main 上 Qwen3.5 的 PD 分离模式适配说明

本文档说明在 **vllm-ascend-main** 仓库中，为 **Qwen3.5** 系列模型接入 **Prefill-Decode（PD）分离部署**需要做的适配。所有路径均为**相对项目根目录（vllm-ascend-main）**的相对路径。

---

## 一、main 分支已具备的 Qwen3.5 支持

| 相对路径 | 内容 |
|----------|------|
| `vllm_ascend/worker/worker.py` | 对 `Qwen3_5ForConditionalGeneration`、`Qwen3_5MoeForConditionalGeneration` 做 KV cache 可用内存 `// 2` |
| `vllm_ascend/spec_decode/eagle_proposer.py` | 多模态列表中包含 `Qwen3_5MoeForConditionalGeneration` |
| `vllm_ascend/quantization/modelslim_config.py` | `qwen3_5_moe` 的 prefix 映射与 packed_modules 配置 |

因此，**PD 场景下 Mooncake 的 KV 传递**与现有 Qwen3 一致：Qwen3.5 使用标准 K/V cache，走「非 MLA / 非 sparse」分支，**无需**修改 `mooncake_connector.py` 或 `mooncake_layerwise_connector.py`。  
`vllm_ascend/ascend_config.py` 中 PD 的 `pd_tp_ratio` / `pd_head_ratio` 在「非 is_deepseek_mla」时按 `num_key_value_heads` 计算，已覆盖 Qwen3.5。

---

## 二、main 分支需要补的适配（推荐）

为在 PD 场景下完整支持 Qwen3.5（含 xlite、EPLB），建议在 main 上做以下两处**代码修改**。

### 2.1 xlite：为 Qwen3.5 架构注册 strategy_map

**目的**：D 节点若开启 xlite 图，加载 Qwen3.5 时不会报 `architecture not supported!`。

**文件**：`vllm_ascend/xlite/xlite.py`

**位置**：`xlite_model_init` 中的 `strategy_map`（约 206–212 行）。

**修改**：在 `strategy_map` 中增加两条（与 Qwen3VL 一样使用 `LlamaXliteModel`）：

```python
strategy_map = {
    "LlamaForCausalLM": LlamaXliteModel,
    "Qwen2ForCausalLM": LlamaXliteModel,
    "Qwen3ForCausalLM": LlamaXliteModel,
    "Qwen3VLForConditionalGeneration": LlamaXliteModel,
    "Qwen3MoeForCausalLM": QwenMoeXliteModel,
    "Qwen3_5ForConditionalGeneration": LlamaXliteModel,
    "Qwen3_5MoeForConditionalGeneration": LlamaXliteModel,
}
```

### 2.2 EPLB：支持 qwen3_5_moe

**目的**：PD + 专家并行时，Qwen3.5 MoE 在 EPLB 中能正确设置 MoE 层数。

**文件**：`vllm_ascend/eplb/utils.py`

**位置**：`config.model_type` 判断分支（约 68–74 行）。

**修改**：在 `qwen3_moe` 分支后增加 `qwen3_5_moe`（逻辑与 `qwen3_moe` 相同）：

```python
if config.model_type == "qwen3_moe":
    model.num_moe_layers = config.num_hidden_layers
elif config.model_type == "qwen3_5_moe":
    model.num_moe_layers = config.num_hidden_layers
elif config.model_type == "deepseek_v2" or config.model_type == "deepseek_v3":
    ...
```

---

## 三、PD 配置与部署（与 Qwen3 一致）

- **vllm 版本**：需使用**已支持 Qwen3.5** 的 vllm（main 或对应发版），并确保 PD 调度会正确设置 `kv_transfer_params`（`do_remote_prefill` / `do_remote_decode`、`remote_block_ids`、`remote_host`、`remote_port`、`remote_engine_id` 等）。
- **connector**：`kv_transfer_config` 中仍使用 `MooncakeConnectorV1`（或逐层场景用 `MooncakeLayerwiseConnector`），无需为 Qwen3.5 单独指定。
- **extra_config**：prefill / decode 的 `tp_size`、`dp_size`、`pp_size` 与现有 Qwen3 的 PD 示例一致；decode 的 `pp_size` 必须为 1。
- **参考文档与示例**（相对路径）：
  - `docs/source/tutorials/pd_disaggregation_mooncake_single_node.md`
  - `docs/source/tutorials/pd_disaggregation_mooncake_multi_node.md`
  - `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md`

---

## 四、适配步骤汇总

1. **确认环境**：vllm 已支持 Qwen3.5；vllm-ascend-main 已拉取最新，并安装 Mooncake、CANN 等依赖。
2. **按第二节修改代码**：  
   - `vllm_ascend/xlite/xlite.py`：增加 Qwen3.5 两种架构的 `strategy_map`。  
   - `vllm_ascend/eplb/utils.py`：增加 `qwen3_5_moe` 分支。
3. **单机验证**：在单机、非 PD 下先跑通 Qwen3.5（及 Qwen3.5-MoE，若需要）。
4. **PD 部署**：P 节点 `kv_role=kv_producer`，D 节点 `kv_role=kv_consumer`，配置与现有 Qwen3 PD 示例一致，仅将模型改为 Qwen3.5。
5. **联调验证**：短序列端到端 P→D 推理，对比单机结果；注意 Mooncake 当前**不支持 IPv6**，P/D 需使用 IPv4。

---

## 五、涉及文件相对路径汇总（main 分支）

| 用途 | 相对路径 |
|------|----------|
| 已支持：Qwen3.5 KV 内存折半 | `vllm_ascend/worker/worker.py` |
| 已支持：Eagle Qwen3.5 MoE | `vllm_ascend/spec_decode/eagle_proposer.py` |
| 已支持：qwen3_5_moe 量化 | `vllm_ascend/quantization/modelslim_config.py` |
| **需改**：xlite Qwen3.5 架构 | `vllm_ascend/xlite/xlite.py` |
| **需改**：EPLB qwen3_5_moe | `vllm_ascend/eplb/utils.py` |
| 无需改：Mooncake 非逐层 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py` |
| 无需改：Mooncake 逐层 | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` |
| 无需改：传输引擎 | `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py` |
| 无需改：PD 头/TP 比例 | `vllm_ascend/ascend_config.py` |
| Connector 注册 | `vllm_ascend/distributed/kv_transfer/__init__.py` |

---

## 六、小结

- **vllm-ascend-main** 已在 worker、eagle_proposer、modelslim 中支持 Qwen3.5；**PD 的 KV 传递逻辑无需改**，按标准 Qwen 类模型即可。
- 为完整支持 Qwen3.5 的 PD 分离，只需在 main 上**补两处**：**xlite 的 strategy_map** 与 **eplb 的 qwen3_5_moe**，然后按现有 Qwen3 的 PD 配置与文档部署即可。

---

*文档基于 vllm-ascend-main 当前目录结构整理；若仓库路径或接口有变更，请以实际代码为准。*
