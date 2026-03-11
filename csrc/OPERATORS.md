# csrc 算子说明文档

本文档描述 vllm-ascend 项目 `csrc` 目录下所有算子及组件的用途。csrc 主要包含 Ascend NPU 上的自定义 CANN/ACLNN 算子、Torch 绑定算子以及构建与工具代码。

---

## 一、目录结构概览

| 目录/文件 | 类型 | 说明 |
|-----------|------|------|
| **aclnn_torch_adapter** | 适配层 | aclnn 与 PyTorch 的 API 适配，用于在 torch_binding 中调用 aclnn 算子 |
| **add_rms_norm_bias** | 自定义 ACLNN 算子 | 带偏置的 Add + RMSNorm 融合 |
| **batch_matmul_transpose** | Torch 侧算子 | 批量矩阵乘 + 转置，支持量化与 ND/NZ 格式 |
| **dispatch_ffn_combine** | 自定义 ACLNN 算子 | MoE FFN 的 dispatch + 分组矩阵乘 + combine 融合（A3） |
| **dispatch_gmm_combine_decode** | 自定义 ACLNN 算子 | decode 阶段 MoE 的 GMM dispatch/combine 融合（A3） |
| **dispatch_layout** | 自定义 ACLNN 算子 | 按 expert 分配结果做 dispatch 布局计算 |
| **grouped_matmul_swiglu_quant_weight_nz_tensor_list** | 自定义 ACLNN 算子 | 分组矩阵乘 + SwiGLU + 量化权重的 NZ 张量列表 |
| **kernels** | Torch 侧内核 | LoRA（bgmv/sgmv）、get_masked_input_and_mask 等 |
| **lightning_indexer** | 自定义 ACLNN 算子 | 稀疏注意力索引（Lightning Attention 稀疏模式） |
| **matmul_allreduce_add_rmsnorm** | 自定义 ACLNN 算子 | 矩阵乘 + AllReduce + Add + RMSNorm 融合（A2） |
| **mla_preprocess** | Torch 侧算子 | MLA（Multi-head Latent Attention）前处理，Q/K/V 与 KV cache 等 |
| **moe_combine_normal** | 自定义 ACLNN 算子 | MoE 多卡下的 combine（按 token 归位） |
| **moe_dispatch_normal** | 自定义 ACLNN 算子 | MoE 多卡下的 dispatch（按 expert 分发 token） |
| **moe_gating_top_k** | 自定义 ACLNN 算子 | MoE 门控 TopK 路由 |
| **moe_init_routing_custom** | 自定义 ACLNN 算子 | MoE 路由初始化与 token 扩展（含量化） |
| **notify_dispatch** | 自定义 ACLNN 算子 | 通信与 dispatch 的同步/通知 |
| **sparse_flash_attention** | 自定义 ACLNN 算子 | 稀疏 Flash Attention |
| **third_party** | 依赖 | 第三方库（如 catlass） |
| **utils** | 工具 | tiling、kernel 公共头文件等 |
| **cmake** | 构建 | CMake 辅助配置 |
| **torch_binding.cpp** | 绑定 | 将上述算子注册到 PyTorch（如 torch.ops） |
| **torch_binding_meta.cpp** | 元实现 | 为 aclgraph 捕获提供 meta 实现 |
| **ops.h** | 声明 | 各算子 C++ 接口声明 |
| **build_aclnn.sh** | 脚本 | 按 SOC 编译自定义 ACLNN 算子并安装 |

---

## 二、自定义 ACLNN 算子详解

以下算子通过 `build_aclnn.sh` 按 SOC（ascend910b / ascend910_93）选择编译，并安装到 `vllm_ascend/_cann_ops_custom`。

### 2.1 add_rms_norm_bias

- **作用**：对两个输入做逐元素相加，再做带可选 bias 的 RMSNorm，输出归一化结果、rstd 以及加和后的中间结果。
- **输入**：x1, x2, gamma, beta（可选）
- **输出**：y（归一化结果）, rstd, x（x1+x2）
- **用途**：Transformer 中 residual add + RMSNorm 的融合，减少访存与 kernel 数。
- **支持 SOC**：ascend910b, ascend910_93。

### 2.2 sparse_flash_attention

- **作用**：基于稀疏索引的 Flash Attention，支持 block_table、query/key RoPE、可变序列长度等。
- **输入**：query, key, value, sparse_indices；可选 block_table、actual_seq_lengths、query_rope、key_rope。
- **输出**：attention_out。
- **属性**：scale_value, sparse_block_size, layout_query/layout_kv, sparse_mode。
- **用途**：长序列/稀疏注意力场景下的高效 attention 计算。
- **支持 SOC**：ascend910b, ascend910_93。

### 2.3 lightning_indexer

- **作用**：根据 query、key、weights 计算稀疏注意力所需的 sparse_indices（Lightning 稀疏模式）。
- **输入**：query, key, weights；可选 actual_seq_lengths、block_table。
- **输出**：sparse_indices。
- **属性**：layout_query, layout_key, sparse_count, sparse_mode。
- **用途**：与 sparse_flash_attention 配合，先算索引再做稀疏 attention。
- **支持 SOC**：ascend910b, ascend910_93。

### 2.4 matmul_allreduce_add_rmsnorm

- **作用**：矩阵乘（x1 @ x2）+ HCCL AllReduce + 与 residual 相加 + RMSNorm，融合为单算子。
- **输入**：x1, x2, residual, gamma。
- **输出**：y, add_out。
- **属性**：group_tp, tp_rank_size, tp_rank_id, epsilon, is_trans_b, is_gather_add_out。
- **用途**：Tensor Parallel 下 MLP 输出的 matmul + allreduce + add + norm 融合（A2）。
- **支持 SOC**：ascend910b。

### 2.5 moe_gating_top_k

- **作用**：MoE 门控：对输入 x（及可选 bias）做 softmax/renorm 等，选出 top-k 个 expert 及对应权重。
- **输入**：x, bias（可选）。
- **输出**：y（权重）, expert_idx, out（可选）。
- **属性**：k, k_group, group_count, group_select_mode, renorm, norm_type, out_flag, routed_scaling_factor, eps。
- **用途**：MoE 模型的路由层，决定每个 token 走哪几个 expert。
- **支持 SOC**：ascend910b, ascend910_93, ascend910_95。

### 2.6 moe_init_routing_custom

- **作用**：根据 gating 得到的 expert_idx 和 scale，对输入 token 做扩展/重排，并输出 expert 维的统计信息（如 token 数、cumsum）。
- **输入**：x, expert_idx, scale（可选）, offset（可选）。
- **输出**：expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum 等。
- **支持**：FP16/BF16/FP32 及 INT8 量化输入。
- **用途**：MoE dispatch 前的准备，为后续按 expert 的矩阵乘提供输入布局。
- **支持 SOC**：ascend910b, ascend910_93。

### 2.7 grouped_matmul_swiglu_quant_weight_nz_tensor_list

- **作用**：多组矩阵乘 + SwiGLU 激活，权重为量化且 NZ 格式的 tensor list；支持 per-group 的 scale。
- **输入**：x, weight（动态 list）, weight_scale, x_scale, group_list。
- **输出**：y, y_scale。
- **用途**：MoE 中多 expert 的 gate/up 投影（量化 W8A16 等）融合计算。
- **支持 SOC**：ascend910b, ascend910_93。

### 2.8 dispatch_ffn_combine（仅 A3）

- **作用**：MoE FFN 的“按 expert dispatch + 分组矩阵乘（w1/w2）+ combine”整段融合，支持 INT8 权重量化。
- **输入**：a（激活）, w1, w2, expertIdx, scale1, scale2, probs。
- **输出**：out。
- **属性**：group（HCCL）, M, transB, weightNz。
- **用途**：A3 上 MoE FFN 的 dispatch-GEMM-combine 一体化，减少通信与 kernel 边界。
- **支持 SOC**：ascend910_93。

### 2.9 dispatch_gmm_combine_decode（仅 A3）

- **作用**：Decode 阶段 MoE 的“按 expert 的 GMM（分组矩阵乘）+ combine”融合，支持量化权重与 expert scale。
- **输入**：x, expert_ids, gmm1_permuted_weight/scale, gmm2_weight/scale, expert_scales 等。
- **输出**：output, expert_token_nums。
- **属性**：group_ep, ep_rank_size/id, moe_expert_num, share_expert 相关, quant_mode, global_bs。
- **用途**：A3 decode 时 MoE 层的高效融合计算与通信。
- **支持 SOC**：ascend910_93。

### 2.10 moe_combine_normal（仅 A3）

- **作用**：MoE 多卡下，将各卡上按 expert 计算完的结果按 token 顺序 combine 回原序（含 HCCL 通信）。
- **输入**：recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tpRecvCountsOptional 等。
- **用途**：Expert Parallel 下 MoE 输出从“按 expert 排布”还原为“按 token 排布”。
- **支持 SOC**：ascend910_93。

### 2.11 moe_dispatch_normal（仅 A3）

- **作用**：MoE 多卡下，根据 topk_idx 和 send/recv offset 将 token 按 expert 分发到各卡（含 HCCL 通信与可选量化）。
- **输入**：x, topk_idx, send_offset, send_tokenIdx, recv_offset, recv_count。
- **输出**：recv_x, x_scales, assist_info_for_combine。
- **属性**：group_ep, ep_world_size, ep_rank_id, group_tp, moe_expert_num, quant_mode, global_bs。
- **用途**：Expert Parallel 下 MoE 的 token-to-expert dispatch。
- **支持 SOC**：ascend910_93。

### 2.12 dispatch_layout（仅 A3）

- **作用**：根据 topk 索引、token 数、expert 数等计算各 rank 各 expert 的 token 布局（numTokensPerRank、numTokensPerExpert、isTokenInRank 等）。
- **用途**：为 MoE dispatch/combine 提供布局信息，供通信与 buffer 分配使用。
- **支持 SOC**：ascend910_93。

### 2.13 notify_dispatch（仅 A3）

- **作用**：在 MoE 的 dispatch 流程中做通信同步/通知（sendData、tokenPerExpertData、sendCount、recvData 等）。
- **用途**：多卡 MoE 时协调各 rank 的发送/接收与数据偏移。
- **支持 SOC**：ascend910_93。

---

## 三、Torch 绑定侧算子与内核

这些算子在 `torch_binding.cpp` 中注册到 PyTorch，不经过 CANN 自定义 op_def，多为 NPU 上的 C++/Kernel 实现或调用 aclnn。

### 3.1 swap_blocks

- **作用**：根据 block_mapping 在 src 与 dst 张量之间按块拷贝数据（用于 KV cache 的 block 重排或拷贝）。
- **用途**：PagedAttention 等场景下的 block 级内存管理。

### 3.2 mla_preprocess

- **作用**：MLA（Multi-head Latent Attention）前处理：对 hidden state 做线性与归一化，生成 Q、更新 KV cache（含 RoPE、slot_mapping）、支持量化与双路输出等。
- **输入**：hiddenState, wdqkv, gamma/beta, wuq, wuk, kv_cache, slotmapping, 各类 scale/offset 等。
- **输出**：q_out0/1, kv_cache_out0/1, inner_out（可选）。
- **用途**：Qwen2-VL MLA、M-RoPE 等结构在 NPU 上的融合前处理。

### 3.3 batch_matmul_transpose

- **作用**：批量矩阵乘 + 转置，支持 ND/NZ 格式与 per_channel/per_token 量化。
- **用途**：批量 GEMM 与 layout 转换的融合。

### 3.4 get_masked_input_and_mask（kernels）

- **作用**：根据 org_vocab / added_vocab 的起止索引与 padding，生成 masked input 与 mask，用于扩展词表或 padding 场景。
- **用途**：LoRA/扩展词表等需要按区间掩码的 embedding 或 logits 处理。

### 3.5 bgmv_shrink / bgmv_expand（kernels）

- **作用**：LoRA 的 BGMV（batched gather-multiply-vector）：shrink 为从大权重中按 indices  gather 并乘 scale 得到低秩矩阵；expand 为将低秩结果写回大矩阵的对应位置。
- **用途**：LoRA 推理时对线性层权重的“收缩-计算-扩展”流程。

### 3.6 sgmv_shrink / sgmv_expand（kernels）

- **作用**：与 BGMV 类似，面向 segment 化序列（seqLen、loraIndices）的 gather-multiply-vector，用于变长序列的 LoRA。
- **用途**：多 batch、变长序列下的 LoRA 计算。

---

## 四、支持与工具目录

### 4.1 aclnn_torch_adapter

- **作用**：提供 aclnn 与 PyTorch 的通用适配（aclTensor 创建、executor 调用等），供 `torch_binding.cpp` 中调用 aclnn 算子（如 MoeCombineNormal、DispatchLayout、NotifyDispatch 等）。

### 4.2 utils

- **作用**：tiling 模板、tiling_type、kernel 公共头（sync_collectives、data_copy、pse、dropmask 等）、fallback、错误码与 aclnn 工具函数，被各自定义算子的 op_host/op_kernel 引用。

### 4.3 third_party

- **作用**：第三方依赖，如 **catlass**（CUTLASS 风格库），被 A2/A3 的 GEMM 类算子（如 grouped_matmul_swiglu、dispatch 等）引用。

### 4.4 cmake

- **作用**：CMake 辅助逻辑，配置 include、编译选项等，供上层 CMakeLists 使用。

---

## 五、SOC 与算子对应关系

| SOC | 说明 | 启用的自定义 ACLNN 算子 |
|-----|------|--------------------------|
| **ascend910b (A2)** | 910B 系列 | grouped_matmul_swiglu_quant_weight_nz_tensor_list, lightning_indexer, sparse_flash_attention, matmul_allreduce_add_rmsnorm, moe_init_routing_custom, moe_gating_top_k, add_rms_norm_bias |
| **ascend910_93 (A3)** | 910C 系列 | 以上全部 + dispatch_ffn_combine, dispatch_gmm_combine_decode, moe_combine_normal, moe_dispatch_normal, dispatch_layout, notify_dispatch |
| **ascend310*** | 310P 等 | 当前无自定义 aclnn 算子，build_aclnn.sh 直接 exit 0 |

---

## 六、如何添加新算子

1. 在 `csrc` 下新建算子目录（如 `my_op`），包含 `op_host`（def、tiling、aclnn 接口）与可选 `op_kernel`（AscendC kernel）。
2. 在 `csrc/build_aclnn.sh` 中按 SOC 将 `my_op` 加入 `CUSTOM_OPS` / `CUSTOM_OPS_ARRAY`。
3. 在 `csrc/torch_binding.cpp` 中调用该算子的 aclnn 接口或自定义 impl，并注册到 `torch.ops`。
4. 在 `csrc/torch_binding_meta.cpp` 中为该算子提供 meta 实现，以便 aclgraph 能正确捕获形状与设备。

更多细节见项目文档：在 `csrc` 目录下创建新操作、在 `build_aclnn.sh` 中配置、在 `torch_binding.cpp` / `torch_binding_meta.cpp` 中绑定。

---

*文档基于当前 csrc 目录结构整理，若算子接口或 SOC 支持有变更，请以代码与 build_aclnn.sh 为准。*
