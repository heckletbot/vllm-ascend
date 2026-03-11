# PD 分离部署中 Mooncake 的 Cache 传递

本文档说明在 Prefill-Decode（PD）分离部署下，Mooncake 如何完成 P 节点到 D 节点的 KV cache 传递，以及相关代码的相对路径与调用关系。

---

## 一、整体架构

- **P 节点（Prefill）**：负责 prefill，算完的 KV cache 需传给 D 节点。
- **D 节点（Decode）**：负责 decode，需从 P 拉取 KV cache 再继续生成。
- **Mooncake**：基于 **Ascend Direct Xfer** 做设备间/节点间 KV cache 的 RDMA 式传输。
- **Connector**：vLLM 侧负责「何时传、传哪些 block、元数据与握手」；实际搬数据由 Mooncake 的 `TransferEngine` 完成。

### 主要文件（相对路径）

| 相对路径 | 作用 |
|----------|------|
| `vllm_ascend/distributed/mooncake_transfer_engine.py` | 封装 Mooncake TransferEngine 单例、内存注册 |
| `vllm_ascend/distributed/mooncake_connector.py` | 非逐层 Connector：整请求一次性传 KV（P→D） |
| `vllm_ascend/distributed/mooncake_layerwise_connector.py` | 逐层 Connector：每算完一层传一层（PCP/DCP 等） |
| `vllm_ascend/distributed/kvpool/backend/mooncake_backend.py` | KV Pool 的 Mooncake 后端（与 connector 配合） |
| `vllm_ascend/distributed/__init__.py` | 注册 `MooncakeConnectorV1`、`MooncakeLayerwiseConnector` |

---

## 二、传输底层：Mooncake TransferEngine

**文件**：`vllm_ascend/distributed/mooncake_transfer_engine.py`

- **GlobalTE**：单例，持有一个 `TransferEngine()`。
- **初始化**：`get_transfer_engine(hostname, device_name)` 中调用  
  `self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)`  
  建立 Mooncake P2P 握手（Ascend 后端）。
- **内存注册**：`register_buffer(ptrs, sizes)` 对 KV cache 设备内存调用 `register_memory(ptr, size)`，供 Mooncake RDMA 访问。
- **限制**：当前 Ascend 后端不支持 IPv6。

实际传输由 connector 中调用 `engine.batch_transfer_sync_read(...)` 完成。

---

## 三、握手与元数据：MooncakeAgentMetadata

P、D 需先交换「对方 KV cache 基址与 RPC 端口」的元数据。

- **MooncakeAgentMetadata** 定义位置：  
  `vllm_ascend/distributed/mooncake_connector.py`（约 58–63 行）
  - 字段：`engine_id`、`te_rpc_port`、`kv_caches_base_addr`、`num_blocks`、`local_ip`。
- **握手通道**：ZMQ（side-channel）
  - 每个 rank 一个 handshake port：`side_channel_port + device_index`。
  - D 向 P 发 `GET_META_MSG`，P 回复编码后的 `MooncakeAgentMetadata`。
  - D 据此得到 P 的 `te_rpc_port` 和 `kv_caches_base_addr`，用于后续 `batch_transfer_sync_read` 的 `session_id = f"{remote_host}:{remote_transfer_port}"`。

---

## 四、非逐层 Connector：整请求传（mooncake_connector.py）

**文件**：`vllm_ascend/distributed/mooncake_connector.py`

### 4.1 角色与线程

- **Scheduler 侧**：`MooncakeConnectorScheduler`（约 910–1095 行）
  - 根据请求的 `kv_transfer_params`（如 `do_remote_prefill` / `do_remote_decode`）维护 `_reqs_need_recv`、`_reqs_need_send`。
  - `build_connector_meta` 将待传请求封装为 `MooncakeConnectorMetadata`（含 `ReqMeta`：local/remote block_ids、remote_host、remote_port 等）。

- **Worker 侧**：`MooncakeConnectorWorker`（约 1098 行起）
  - **kv_producer（P）**：启动 **KVCacheSendingThread**。
  - **kv_consumer（D）**：启动 **KVCacheRecvingThread**。
  - 二者均调用 `global_te.get_transfer_engine(...)` 与 `global_te.register_buffer(ptrs, lengths)`。

### 4.2 P 节点：发送与延迟释放

| 功能 | 相对路径 | 位置（约） |
|------|----------|------------|
| 发送线程主循环（GET_META / DONE_RECVING） | `vllm_ascend/distributed/mooncake_connector.py` | `KVCacheSendingThread` 172–293 行，`run_busy_loop` |
| 请求结束且需传给 D 时延迟 free | `vllm_ascend/distributed/mooncake_connector.py` | `MooncakeConnectorScheduler.request_finished` 1046–1081 行 |
| P Worker 注册 KV 并启动发送线程 | `vllm_ascend/distributed/mooncake_connector.py` | `MooncakeConnectorWorker.register_kv_caches` 1202–1302 行 |

### 4.3 D 节点：接收与 Mooncake 传输

| 功能 | 相对路径 | 位置（约） |
|------|----------|------------|
| 接收队列与循环 | `vllm_ascend/distributed/mooncake_connector.py` | `KVCacheRecvingThread.run` / `_handle_request` 406–451 行 |
| **Mooncake 实际传输**（batch_transfer_sync_read） | `vllm_ascend/distributed/mooncake_connector.py` | `KVCacheRecvingThread._transfer_kv_cache` 468–526 行 |
| 向 P 拉取元数据（GET_META_MSG） | `vllm_ascend/distributed/mooncake_connector.py` | `KVCacheRecvingThread._get_remote_metadata` 579–700 行 |
| 收完后通知 P（DONE_RECVING_MSG） | `vllm_ascend/distributed/mooncake_connector.py` | `_send_done_recv_signal` 702–738 行 |
| Scheduler 决定哪些请求要收 | `vllm_ascend/distributed/mooncake_connector.py` | `MooncakeConnectorScheduler.update_state_after_alloc` 884–912 行 |
| D Worker 发起拉取任务 | `vllm_ascend/distributed/mooncake_connector.py` | `MooncakeConnectorWorker.start_load_kv` 1568–1644 行 |

### 4.4 元数据与 Connector 接口

| 功能 | 相对路径 | 位置（约） |
|------|----------|------------|
| ReqMeta / MooncakeConnectorMetadata | `vllm_ascend/distributed/mooncake_connector.py` | 66–76 行、772–798 行 |
| MooncakeConnector 对外接口（get_num_new_matched_tokens、build_connector_meta 等） | `vllm_ascend/distributed/mooncake_connector.py` | `MooncakeConnector` 801–909 行 |

---

## 五、逐层 Connector（mooncake_layerwise_connector.py）

**文件**：`vllm_ascend/distributed/mooncake_layerwise_connector.py`

- **思路**：每算完一层传一层，D 可边收边算。
- **发送**：P 在 `save_kv_layer`（约 1088–1145 行）中按层准备 block，必要时做 `kv_alltoall_and_rearrange`，由 **KVCacheSendingLayerThread** 按层发送。
- **接收**：**KVCacheRecvingLayerThread** 按层接收，收齐一层后 D 使用该层 decode。
- **元数据**：`MooncakeLayerwiseConnectorMetadata`、`MooncakeAgentMetadata`（约 51–67 行、422–446 行）。

| 功能 | 相对路径 | 位置（约） |
|------|----------|------------|
| 逐层发送线程 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | `KVCacheSendingLayerThread` |
| 逐层接收线程 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | `KVCacheRecvingLayerThread` |
| 按层保存并触发发送 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | `save_kv_layer` 1088–1145 行 |
| Layerwise Connector 入口 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | `MooncakeLayerwiseConnector` 448 行起 |

---

## 六、数据流概览（非逐层）

```
1. 配置与初始化
   - kv_transfer_config 指定 connector 为 MooncakeConnectorV1
   - P/D 启动时：Worker 中 get_transfer_engine()、register_buffer(kv_caches 的 ptrs/lengths)
   - P 启动 KVCacheSendingThread（ZMQ 监听）；D 启动 KVCacheRecvingThread（消费 request_queue）

2. P 上 prefill 完成且需传给 D
   - Scheduler：request_finished 发现 do_remote_decode 且 FINISHED_LENGTH_CAPPED
     → 不立刻 free block，写入 _reqs_need_send，build_connector_meta 时把 remote_block_ids、remote_host、remote_port 等带给 D。

3. D 需要该请求的 KV
   - Scheduler：get_num_new_matched_tokens / update_state_after_alloc 将请求加入 _reqs_need_recv，build_connector_meta 生成 ReqMeta。
   - Worker：start_load_kv(metadata) 对每个 ReqMeta 调用 kv_recv_thread.add_request(...)。

4. D 的 KVCacheRecvingThread
   - _get_remote_metadata：ZMQ 向 P 发 GET_META_MSG，拿到 MooncakeAgentMetadata。
   - _transfer_kv_cache：构造 src_list、dst_list、length_list，调用 engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)。
   - 可选：reformat_kv_cache（concat / NZ）。
   - _send_done_recv_signal：ZMQ 发 DONE_RECVING_MSG 给 P。

5. P 的 KVCacheSendingThread
   - 收到 DONE_RECVING_MSG 后 update_done_task_count，超时或收齐后释放该请求的 block（delayed_free_requests）。
```

---

## 七、关键代码位置速查（相对路径）

| 功能 | 相对路径 | 行号/说明 |
|------|----------|-----------|
| TransferEngine 初始化与内存注册 | `vllm_ascend/distributed/mooncake_transfer_engine.py` | GlobalTE：get_transfer_engine、register_buffer |
| P 元数据与 DONE_RECVING 处理 | `vllm_ascend/distributed/mooncake_connector.py` | KVCacheSendingThread.run_busy_loop，GET_META_MSG / DONE_RECVING_MSG |
| D 拉取 KV（Mooncake 实际传输） | `vllm_ascend/distributed/mooncake_connector.py` | KVCacheRecvingThread._transfer_kv_cache，engine.batch_transfer_sync_read |
| D 向 P 要元数据 | `vllm_ascend/distributed/mooncake_connector.py` | KVCacheRecvingThread._get_remote_metadata |
| D 收完后通知 P | `vllm_ascend/distributed/mooncake_connector.py` | _send_done_recv_signal，DONE_RECVING_MSG |
| Scheduler 决定收/发请求 | `vllm_ascend/distributed/mooncake_connector.py` | MooncakeConnectorScheduler：update_state_after_alloc、request_finished、build_connector_meta |
| D Worker 发起拉取 | `vllm_ascend/distributed/mooncake_connector.py` | MooncakeConnectorWorker.start_load_kv，add_request 到 kv_recv_thread |
| 逐层发送 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | save_kv_layer，KVCacheSendingLayerThread |
| 逐层接收 | `vllm_ascend/distributed/mooncake_layerwise_connector.py` | KVCacheRecvingLayerThread，start_load_kv |
| Connector 注册 | `vllm_ascend/distributed/__init__.py` | MooncakeConnectorV1、MooncakeLayerwiseConnector |

---

## 八、相关测试与工具（相对路径）

| 用途 | 相对路径 |
|------|----------|
| Mooncake Connector 单测 | `tests/ut/kv_connector/test_mooncake_connector.py` |
| Mooncake Layerwise Connector 单测 | `tests/ut/kv_connector/test_mooncake_layerwise_connector.py` |
| Mooncake 依赖安装脚本 | `tools/mooncake_installer.sh` |
| PD 部署示例与说明 | `examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md` |

---

*文档基于 vllm-ascend 当前代码整理，行号与接口若有变更请以仓库为准。*
