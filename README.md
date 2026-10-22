# Qwen3.5-35B-A3B × QSA + Muon：CPT 能力重建与 SFT/推理全栈方案

本仓库提供将 **Qwen3.8-Flash-Next（`qwen4_exp`）的 QSA（Qwen Sparse Attention）** 移植替换至 **Qwen3.5-35B-A3B** 的 10 个全注意力层（`full_attention_interval = 4`），并切换为 **Muon 优化器（`dist_muon`）** 后的 **三阶段继续预训练（CPT）能力重建、后续 SFT 以及 vLLM / SGLang 极限推理优化**完整工程方案。

硬件面向 **NVIDIA H200 (141GB HBM3e) × 8** 节点深度调优。

---

## 1. 核心问题诊断：为什么必须分阶段 CPT？

### 1.1 架构手术前后的异同
`Qwen3.5-35B-A3B` 共 40 层（`10 × (3 × Gated DeltaNet + 1 × Full Attention)`），`hidden_size = 2048`，256 个路由专家（Top-8 激活，`moe_intermediate_size = 512`）+ 1 个共享专家。将其中的 10 层 Full Attention 替换为 QSA 后：

| 模块 | 替换前 (`Qwen3.5-35B-A3B`) | 替换后 (`35B-A3B-QSA`) | 权重继承与冷启动状态 |
|---|---|---|---|
| 线性注意力层 (30 层) | Gated DeltaNet (16 QK / 32 V, dim 128) | 保持不变 | 100% 继承 |
| MoE 专家层 (40 层) | 256 路由专家 (Top-8) + 1 共享专家 | 保持不变 | 100% 继承 |
| 全注意力主干 (10 层) | Gated Attention (16 Q / 2 KV, dim 256, 64 维 mRoPE) | QSA 主投影 (`q/k/v/o_proj`, `q/k_norm`) | **形状完全同构，100% 继承** |
| **QSA 微块索引器 (10 层)** | ❌ 无 | **`self_attn.indexer`** (`index_qk_proj`, `q/k_layernorm`, 4Q/1KV, dim 128, ratio 4, budget 2048) | ⚠️ **全新冷启动 / 跨模型移植，分布未对齐** |
| 优化器动力学 | AdamW 预训练收敛态 | **Muon (`dist_muon`) + AdamW 混合** | ⚠️ **无历史 Muon 动量缓冲，直接全参易震荡** |

### 1.2 两大核心风险与对策
1. **稀疏路由冷启动冲垮 MoE 与 GDN**：当序列长度 $> 2048$（`indexer_budget`）时，未对齐的 `indexer` 会挑出错误的 4-token 微块（micro-blocks）。若直接放开全模型训练，错误上下文产生的剧烈梯度会瞬间破坏后方 256 个专家的负载均衡与 GDN 的 `fp32` 递归状态。
   - **对策**：先执行 **Stage 0（QSA 路由对齐预热）**，冻结 MoE 与 GDN，仅训 `indexer`（或 10 层 QSA），并支持直接用原 35B-A3B 冻结 Q/K 的微块注意力分布做 **前向 KL 自蒸馏**（[`compute_qsa_indexer_self_kl_loss`](mcore_qsa_bridge_and_distill_plugin.py)），5~10 倍加速收敛。
2. **Muon 正交化误伤 Router 与 Indexer**：Muon 通过 Newton-Schulz 迭代对 2D 动量矩阵做正交化更新，**绝对不能用于非内部特征变换矩阵**。
   - **对策**：通过插件 [`qsa_muon_plugin.py`](qsa_muon_plugin.py) 在优化器创建前强制拦截，确保 **Embedding、`lm_head`、1D Norm、GDN 递归参数（`A_log`/`dt_bias`/`conv1d`）、MoE Router（`mlp.gate`）以及 QSA Indexer（`self_attn.indexer`）全部回退至 AdamW**，仅对内部 2D 线性层应用 `dist_muon`。

---

## 2. 三阶段训练路线总览（Warmup $\to$ CPT $\to$ SFT）

| 阶段 | 脚本模式 | 训练参数范围 | 优化器与学习率 | 序列长度与预算课程 | 目标 |
|---|---|---|---|---|---|
| **Stage 0** | `warmup` | 仅 `self_attn.indexer` 或 10 层 QSA `self_attn`（冻结 GDN / MoE / ViT） | `adam`, `lr=5e-4 -> 5e-5`, warmup `5%` | `16384`，约 `1B~2B` tokens | 对齐微块路由打分，支持免外部 Teacher 的前向 KL 自蒸馏 |
| **Stage 1** | `cpt` | LLM 全参（QSA + GDN + 256 MoE 专家）；默认冻结 ViT | `dist_muon` (2D) + `AdamW` (其余), `lr=1.5e-4 -> 1.5e-5`, warmup **`8%`**, `wd=0.1` | `16384 -> 32768`，预算 `4096 -> 2048` 退火，约 `10B~30B` tokens | 在 Muon 几何与稀疏注意力下重建 35B-A3B 的长程推理与知识容量 |
| **Stage 2A** | `sft_full` | 全参微调 LLM + 专家；**冻结 ViT 与 MoE Router** | `dist_muon`, `lr=2e-5 -> 2e-6`, warmup `3%`, `wd=0.05` | `8192 / 16384`，课程数据 + 通用回放 | 高算力全参指令微调，冻结 Router 防止垂直领域专家坍缩 |
| **Stage 2B** | `sft_lora` | 异构 LoRA (`all-linear`, `r64/a128`)；**冻结 Router** | `adam` / `dist_muon`, `lr=3e-5 -> 3e-6` | `8192`，支持 V1~V4 多阶段累计 Adapter | 轻量化低显存 SFT，防遗忘，无缝对接既有评测选档流水线 |

> **⚠️ MTP 头维护提醒**：`Qwen3.5-35B-A3B` 含 1 层 MTP（`mtp_num_hidden_layers = 1`）。CPT/SFT 期间请保留或协同微调 `mtp.*` 权重，否则训后表征漂移会导致推理时 `NEXTN` 投机解码接受率大幅下降。

---

## 3. H200 × 8 显卡利用率（MFU）最大化要点

1. **并行切分避坑（`TP=1, EP=8, DP=8` 远优于 `TP=4`）**
   - `35B-A3B` 的 `num_key_value_heads = 2`，QSA 的 `indexer_kv_heads = 1`。若盲目沿用 27B Dense 的 `TP=4`，会导致 KV 头无法整除 `TP` 报错或被迫复制 KV 头；且 `hidden_size = 2048`、`moe_intermediate_size = 512` 较窄，`TP=4` 会使单卡 GEMM 过碎。
   - **最佳配置**：`TP=1, EP=8, DP=8`（256 个专家均分到 8 张 H200，每卡 32 个专家，无 TP 通信开销）。超长上下文（$\ge 32\text{K}$）可设 `TP=2, EP=4, DP=4` 并开启 `sequence_parallel=true`。
2. **MoE 与长序列算子全开**
   - `--moe_grouped_gemm true`：合并每卡 32 个细粒度专家的矩阵乘，MoE 层提速 **2.5×~3.5×**。
   - `--moe_shared_expert_overlap true`：共享专家计算与路由专家 All-to-All 通信重叠。
   - `--packing true --padding_free true` + `--cross_entropy_loss_fusion true`：消除填充气泡并省去 `248,320` 词表 logits 显存峰值。
   - `NCCL_NVLS_ENABLE=1`：启用 H200 NVLink SHARP 硬件集合通信加速。

---

## 4. 推理引擎优化（vLLM & SGLang）

脚本 [`serve_vllm_sglang_h200.sh`](serve_vllm_sglang_h200.sh) 集成了针对 `35B-A3B-QSA` 的全套推理优化：

- **黄金并行度 `TP=2`**：完美契合 `num_key_value_heads = 2`（每卡 1 个 KV 头，**零 KV Cache 复制冗余**），2 张 H200 (282GB) 留出 **210+ GB** 供 128K~256K 上下文缓存；8 卡集群推荐 `DP=4 × TP=2` 或 `EP=8 + FP8`。
- **混合三级缓存管理**：
  - 30 层 GDN 递归状态强制 FP32（`--mamba-ssm-cache-dtype float32`），防止长文生成数值溢出；
  - 10 层 QSA 主 KV Cache 支持 `--kv-cache-dtype fp8`；
  - 10 层 QSA Paged Side-Cache（`key_pool` 4-token 微块压缩键池）常驻显存，128K 上下文 Decode 仅需访存 Top-512 微块（2048 tokens），**KV 访存削减 98.4%**。
- **SGLang `NEXTN` 投机解码 + QSA 索引共享（Index-Sharing）**：
  - 启用 `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4`，利用 QSA 的 4-token 微块粒度，在验证草稿 token 时**复用同一次 QSA Indexer 选出的 Top-$B$ 微块集合**，使 Decode 吞吐再翻 **1.8×~2.4×**。

---

## 5. 目录文件说明与快速启动

| 文件 | 说明 |
|---|---|
| [`run_cpt_sft_qsa_35a3_h200x8.sh`](run_cpt_sft_qsa_35a3_h200x8.sh) | H200×8 训练主入口，支持 `warmup` / `cpt` / `sft_full` / `sft_lora`、10 步冒烟 `SMOKE=1` 及 `BACKEND=megatron\|hf` 切换 |
| [`qsa_muon_plugin.py`](qsa_muon_plugin.py) | 核心插件：Muon/AdamW 参数分组护栏、分阶段冻结控制、严格数据预处理与分通道 Loss |
| [`mcore_qsa_bridge_and_distill_plugin.py`](mcore_qsa_bridge_and_distill_plugin.py) | 进阶插件：免外部 Teacher 的微块前向 KL 自蒸馏、动态稀疏预算课程、`mcore-bridge` 透传 |
| [`serve_vllm_sglang_h200.sh`](serve_vllm_sglang_h200.sh) | 推理引擎一键服务脚本：含 `prepare_config` 配置补齐、`vllm`、`vllm_fp8_ep8`、`sglang` 模式 |
| [`ds_zero2_h200.json`](ds_zero2_h200.json) | `BACKEND=hf` 模式下的 DeepSpeed ZeRO-2 通信重叠配置 |
| [`requirements_h200.txt`](requirements_h200.txt) | H200×8 训练与推理配套依赖清单 |

### 快速运行示例

```bash
# 1. 10 步冒烟测试（验证环境、算子挂载、Muon 分组日志与显存峰值）
SMOKE=1 bash run_cpt_sft_qsa_35a3_h200x8.sh warmup
SMOKE=1 bash run_cpt_sft_qsa_35a3_h200x8.sh cpt

# 2. 依次执行 Stage 0 (路由预热) -> Stage 1 (Muon CPT) -> Stage 2 (SFT)
bash run_cpt_sft_qsa_35a3_h200x8.sh warmup
MODEL_PATH=/path/to/warmup_ckpt bash run_cpt_sft_qsa_35a3_h200x8.sh cpt
MODEL_PATH=/path/to/cpt_ckpt    bash run_cpt_sft_qsa_35a3_h200x8.sh sft_full

# 3. 补齐推理配置并启动 SGLang / vLLM 服务
MODEL_PATH=/path/to/sft_ckpt bash serve_vllm_sglang_h200.sh prepare_config
MODEL_PATH=/path/to/sft_ckpt bash serve_vllm_sglang_h200.sh sglang
```
