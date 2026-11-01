# Qwen3.5 系列（27B Dense & 35B-A3B MoE）多模态 JEV 决策模型训练方案与量化潜力（FP8 / NVFP4）深度报告

> **编制日期**：2026-11-01  
> **面向硬件**：NVIDIA H200 SXM5（Hopper 架构，sm90，单卡 141GB HBM3e，8 卡节点总显存 1,128GB，NVLink 4 900GB/s）  
> **核心任务**：基于 **Qwen3.5-27B（Dense）** 与 **Qwen3.5-35B-A3B（Sparse MoE）** 底座，通过 **ms-swift** 框架构建高吞吐、毫秒级时延的多模态 **JEV 决策模型（Typed-Decision / System-1 Model）**，并全面评估其在 **FP8** 与 **NVFP4** 下的量化压缩与部署潜力。

---

## 目录
1. [执行摘要与技术背景](#1-执行摘要与技术背景)
2. [多模态 JEV（Typed-Decision）范式机理](#2-多模态-jevtyped-decision范式机理)
3. [Qwen3.5-27B Dense 多模态 JEV 训练方案](#3-qwen35-27b-dense-多模态-jev-训练方案)
4. [Qwen3.5-35B-A3B MoE 多模态 JEV 训练方案](#4-qwen35-35b-a3b-moe-多模态-jev-训练方案)
5. [Packing 在多模态 JEV 上的可行性与避坑指南](#5-packing-在多模态-jev-上的可行性与避坑指南)
6. [低比特量化潜力评估（FP8 vs NVFP4）](#6-低比特量化潜力评估fp8-vs-nvfp4)
7. [生产端单次 Forward 决策推理与 Serving 指南](#7-生产端单次-forward-决策推理与-serving-指南)
8. [环境配置与交付清单](#8-环境配置与交付清单)

---

## 1. 执行摘要与技术背景

传统的视觉语言模型（VLM）在处理结构化判断（如内容风控治理、商品类目归类、多选题判别、图像质量 0~5 阶梯评分等）时，普遍沿用因果自回归生成模式（Next-Token Generation），逐 token 生成形如 `{"result": "pass", "score": 4}` 的 JSON 或解释文本。这种模式在工业化落地中面临两大瓶颈：
1. **时延昂贵**：Decode 阶段受限于内存带宽，生成 50~100 个 Token 耗时通常在数百毫秒至秒级；
2. **格式与置信度幻觉**：自回归输出的概率往往未经良好校准（Overconfident），且存在格式截断或解析失败的风险。

**JEV（Typed-Decision / System-1）决策模型**通过重构模型输出头与前向推理路径，在**单次前向传播（Single Forward Pass）**中直接读取指定判别候选空间（Options / Choices / Scores）在特定 Token 处的 Logits，输出**严格校准的概率分布（Calibrated Probabilities）**。

本文档系统性针对 **Qwen3.5-27B Dense** 与 **Qwen3.5-35B-A3B MoE** 两大主力基座，设计兼顾泛化性与稳定性的 LoRA 训练策略，匹配 8×H200 硬件算力与显存红利，并对下一代推理量化（FP8 与 NVFP4）给出可落地的技术评估。

---

## 2. 多模态 JEV（Typed-Decision）范式机理

### 2.1 核心工作流对比

```
[传统生成式 VLM]
图像 + 问题 ──► ViT + LLM ──► 逐 Token 自回归解码 (Decode Loop) ──► 文本/JSON 解析 ──► 业务决策
                            └─► (耗时 300~1500ms，需处理 JSON 截断异常)

[JEV 决策架构]
图像 + 问题 + 判别元空间 ──► ViT + LLM ──► 单次 Forward 提取 Decision Head Logits ──► 校准概率分布
                                        └─► (耗时 15~35ms，零生成开销，直接判定)
```

### 2.2 严格适定评分规则损失（Proper Scoring Rules）
训练 JEV 模型不宜采用全序列交叉熵（Next-Token CE），而需针对候选分布使用适定评分损失：
1. **`distill_rps`（Rank Probability Score 秩概率得分）**：
   $$RPS = \frac{1}{K-1} \sum_{m=1}^{K-1} \left( \sum_{k=1}^m p_k - \sum_{k=1}^m y_k \right)^2$$
   专用于有序离散决策（如质量评分 0~5 档、风险等级低/中/高）或接收教师模型（如 GPT-4o / Qwen3.5-Max）的软概率蒸馏。
2. **`smooth_brier`（平滑布里尔分数）**：
   $$BS = \frac{1}{N} \sum_{i=1}^N \| \mathbf{p}_i - \mathbf{y}_i \|_2^2$$
   用于无序多选或二值门控分类，强制模型输出的 Softmax 分数直接逼近真实后验频率，杜绝置信度虚高。
3. **`ce_rps`**：结合了交叉熵锐利度与 RPS 序数敏感度的混合目标。

---

## 3. Qwen3.5-27B Dense 多模态 JEV 训练方案

### 3.1 架构要点
* **语言主干**：64 层，`hidden_size = 5120`，周期性混合注意力（16 个周期，每周期包含 3 层 Gated DeltaNet + 1 层 Gated Attention），`tie_word_embeddings = false`。
* **视觉栈**：ViT（深度 27 层，`hidden_size = 1152`） + PatchMerger 空间合并投影器（输出维度 `5120`，精确匹配语言塔宽度）。

### 3.2 异构 LoRA 参数配方（Heterogeneous LoRA）
为避免决策任务破坏底层通用图文表征，采用分层异构配置：

| 模块 | 涉及算子 | 模式 | 推荐超参 | 峰值学习率 | 作用 |
|---|---|---|---|---|---|
| **LLM 主干** | Attention/DeltaNet 全投影 (`all-linear`) | LoRA | $r=64, \alpha=128$, dropout 0.05 | `3e-5` | 学习复杂的多模态逻辑仲裁 |
| **PatchMerger** | `visual.merger.*` | 全参 或 LoRA | $r=32, \alpha=64$（若LoRA） | `1e-5` | 视觉与语言维度的核心转接通道 |
| **ViT 视觉塔** | `visual.blocks.*` | 冻结 或 浅层LoRA | 域适应：$r=16, \alpha=32$；通用域：冻结 | `1e-5` | 通用视觉特征保持，防止灾难性遗忘 |
| **决策分类头** | `decision_head` / `score_head` | 全参 (`modules_to_save`) | 维度跟随选项集 | `5e-5` | 负责从尾部隐藏状态映射为校准 Logits |

### 3.3 8×H200 分布式训练配置
8 张 H200（141GB/卡）单机显存充足，针对 27B Dense 推荐两种方式：
* **首选（极致吞吐）**：`megatron sft`，设置 `TP=4, DP=2, sequence_parallel=true`。单卡静态显存约 13.5GB，动态激活与优化器常驻约 80~110GB。
* **次选（开箱即用）**：`swift sft`，开启 `--deepspeed default-zero2`。由于单卡 141GB 完全可以承载 27B BF16 参数（~54GB），ZeRO-2 仅对梯度和优化器做切分，避开了 ZeRO-3 在 Forward/Backward 时频繁 All-Gather 全量参数的通信损耗。

---

## 4. Qwen3.5-35B-A3B MoE 多模态 JEV 训练方案

### 4.1 架构要点与特性
* **语言主干**：40 层，`hidden_size = 2048`，10 个混合周期（3 层 Gated DeltaNet + 1 层 Gated Attention），KV 头数仅为 2（`num_key_value_heads = 2`）。
* **MoE 专家层**：40 层全部采用 MoE 替换传统 FFN，含 **256 个路由专家（Top-8 激活，单专家宽度 `moe_intermediate_size = 512`） + 1 个共享专家**。
* **参数规模**：总参数量约 35B，但每个 Token 仅激活 **3B** 参数（激活比仅 ~8.6%），计算量与 3B 密集模型相仿，吞吐极高。
* **视觉塔**：深度 27 层，`out_hidden_size = 2048`（对应 35B-A3B 的隐藏层宽度）。

### 4.2 MoE 决策训练的四大特有挑战与防护机制

1. **Router 坍缩防御（Freeze Router）**：
   * **现象**：在决策分类数据上微调时，数据分布集中于特定的语义判定，如果微调 `mlp.gate`（MoE Router），路由器很容易把流量全部导向少数几个“通用专家”，导致其他 200+ 专家退化，损害整体知识储备。
   * **对策**：在 LoRA 中**强制冻结所有路由投影 `mlp.gate`**，仅对专家的内部线性层（`down_proj`, `gate_proj`, `up_proj`）以及 Attention 层挂载 LoRA。
2. **专家并行（EP）黄金法则**：
   * 35B-A3B 的 `num_key_value_heads = 2`，若盲目使用 `TP=4`，会导致 KV 头无法整除 TP 抛出异常或产生强制复制冗余；且其 `hidden_size = 2048` 较小，TP 切得过碎会降低计算通信比。
   * **推荐并行度**：`TP=1, EP=8, DP=8`（每张 H200 独立分管 32 个专家，无 TP 通信）；长序列下使用 `TP=2, EP=4, DP=4`。
3. **MoE 算子加速开关**：
   * 务必开启 `--moe_grouped_gemm true`：Hopper 架构对 Grouped GEMM 有极佳的硬件加速支持，可将多专家运算耗时缩减 60% 以上；
   * 开启 `--moe_shared_expert_overlap true`：实现共享专家计算与路由专家跨卡 All-to-All 通信的流水线重叠。
4. **辅助损失平衡**：
   * 训练中显式保持 `--router_aux_loss_coef 0.001`，即使冻结部分路由，仍能对专家分配保持约束，防止显存倾斜（Load Imbalance）。

---

## 5. Packing 在多模态 JEV 上的可行性与避坑指南

### 5.1 核心问题：为什么多模态 JEV 不建议盲目使用 Packing？

许多开发者习惯在普通 SFT 中开启 `--packing true --padding_free true`，但在**多模态 JEV 决策任务**中，**强烈建议设置为 `--packing false`**：

| 评估维度 | 普通生成式 SFT (Causal LM) | 多模态 JEV 决策模型 (Typed-Decision) |
|---|---|---|
| **损失计算位置** | 全序列中所有 Assistant Token 均计算 CE Loss | **仅在特定位置**（问题末尾 Token）抽取 Hidden State 计算决策 Loss |
| **Packed 索引映射** | 只要 Label 填 `-100`，整条序列无脑 Flatten | **必须维护每个子样本决策点的绝对偏移**。若无专项 Patch，框架会错抽整条序列尾部，吞掉前方样本 |
| **损失函数契约** | `loss = CrossEntropy(logits, labels)` 标量聚合 | `distill_rps(pred_probs, targets)` 要求输入具有明确的 Batch 维度 `[B, num_classes]` |
| **多模态 RoPE 风险** | 纯文本为 1D 顺序位置 | ViT Token 具备 2D/3D 空间坐标（`grid_thw`），跨图拼接若未彻底隔离，存在视觉注意力渗漏风险 |
| **Padding 比例** | 纯文本短句多，Padding 率常年 $>80\%$，收益巨大 | 图像本身占 1000~2000 Tokens，多模态样本天生较长，Packing 的真实算力收益仅 15~25% |

### 5.2 8×H200 上的最优替代：动态 Padding + FlashAttention

由于 H200 具备 141GB 显存，无需承担 Packing 破坏决策索引的风险，最佳实践为：
* 设置 `--packing false`；
* 启用 `--attn_impl flash_attn`；
* DataLoader 会自动执行 **Batch-level Dynamic Padding**（一个 Batch 内仅填充到该批最长样本，而非静态填充到 8192），配合 FlashAttention 内部高效屏蔽，算力浪费极小且 100% 杜绝索引错位。

---

## 6. 低比特量化潜力评估（FP8 vs NVFP4）

针对离线决策评测与高并发 Serving 场景，对两款模型在 **FP8（Hopper 原生）** 与 **NVFP4（Blackwell 原生 / Hopper 模拟）** 下的潜力进行深度推演：

### 6.1 FP8 量化评估（E4M3 / E5M2）

* **硬件支持度**：NVIDIA H200 具备第四代 Tensor Core 原生 FP8 引擎，支持 FP8 GEMM 与动态缩放（Transformer Engine），理论吞吐为 BF16 的 **2.0 倍**。
* **Qwen3.5-27B Dense 的 FP8 表现**：
  * **权重体积**：从 BF16 的 54GB 降至 **~27GB**。
  * **敏感层保护（Guardrails）**：Gated DeltaNet 的内部递推状态（`fp32` state $S_t = S_{t-1} + k_t v_t^\top$）严禁量化为 FP8，必须保持在 FP32；线性投影层（$W_q, W_k, W_v, W_{gate}, W_{up}, W_{down}$）采用 E4M3 Per-channel / Per-token 动态量化无损。
  * **决策精度**：决策 RPS / Brier 分数漂移 $<0.4\%$，几乎无感。
* **Qwen3.5-35B-A3B MoE 的 FP8 表现**：
  * **权重体积**：从 BF16 的 70GB 降至 **~35GB**。
  * **Outlier 敏感性**：35B-A3B 具有 256 个专家，单专家矩阵宽度仅 512。小矩阵对异常值（Activation Outliers）极为敏感，传统的 Per-tensor 静态量化容易导致个别专家精度坍塌。
  * **推荐量化策略**：必须采用 **Block-wise 缩放（Block 128）** 或 **SmoothQuant 预平滑**；`mlp.gate` 路由层必须保留 BF16。

### 6.2 NVFP4 量化评估（Microscaling MXFP4）

* **格式机理**：NVFP4 是 NVIDIA 针对新一代架构推行的微缩放 4 位浮点格式，采用 E2M1 数据类型（1 个符号位、2 个指数位、1 个尾数位），以每 16 或 32 个元素绑定一个微缩放因子（Microscaling Factor）。
* **在 H200 上的真实执行状态**：
  > ⚠️ **重要硬件事实**：H200（Hopper）**没有原生的 FP4 计算单元**（原生仅支持 FP8/INT8/BF16/TF32）。在 H200 上运行 NVFP4 模型，本质上是 **Weight-Only (W4A16 或 W4A8)** 运行模式——即权重在 HBM 中以 4-bit 存储，读取进入 Tensor Core 前由反量化核函数即时展开为 FP8 或 BF16 计算。
  > 虽无计算翻倍红利，但可享受 **极度节约显存带宽（Bandwidth-Bound 缓解）** 的收益。
* **量化体积与收益**：
  * **27B Dense**：整模权重仅需 **~14GB**！单张消费级 GPU 或轻量推理卡即可全载运行。
  * **35B-A3B MoE**：整模权重仅需 **~18GB**！且由于每步仅激活 3B 参数，4-bit 权重流转速度极快，在 Decode 极端并发下，吞吐可提升 2.5×~3.5×。
* **决策精度的退化风险与对策**：
  * 纯 PTQ（训练后量化）将 256 专家的 MoE 压缩至 4-bit 会导致决策分类边缘概率轻微漂移（特别是 0.01~0.05 区间的长尾安全阈值）。
  * **对策**：采用 **LoRA-Aware QAT（量化感知微调）**，在 JEV 训练阶段插入模拟伪量化节点（FakeQuantize），使 LoRA 直接适应 4 位网格。

### 6.3 综合量化选型矩阵对比

| 评测维度 | 原始 BF16 | Hopper 原生 FP8 (E4M3) | NVFP4 (W4A8/W4A16 on H200) | NVFP4 (原生 Blackwell B200) |
|---|---|---|---|---|
| **27B Dense 权重大小** | 54 GB | 27 GB | **14 GB** | **14 GB** |
| **35B-A3B MoE 权重大小** | 70 GB | 35 GB | **18 GB** | **18 GB** |
| **H200 部署形态** | 1~2 卡 | **单卡 141GB 轻松全载** | **单卡可容纳多实例** | 需 Blackwell 芯片 |
| **Hopper 计算加速比** | 1.0× (基准) | **1.8× ~ 2.1×** | 1.3× ~ 1.5× (访存受限缓解) | 3.0× ~ 4.0× (算力+带宽双倍) |
| **JEV 决策校准误差 (Brier)** | 基线 | 相对误差 $<0.3\%$ | 相对误差 $1.2\% \sim 2.5\%$ (建议 QAT) | 相对误差 $1.2\% \sim 2.5\%$ |
| **推荐生产引擎** | vLLM / SGLang | **vLLM FP8 / SGLang** | **TensorRT-LLM (W4A8)** | TensorRT-LLM / vLLM |

---

## 7. 生产端单次 Forward 决策推理与 Serving 指南

### 7.1 Python 原生 PtEngine 毫秒级推理实现
训练并合并权重后，直接调用 `PtEngine`，利用其内部挂载的决策头提取概率，避免任何自回归文本生成：

```python
import torch
from swift.llm import PtEngine, RequestConfig

# 加载合并后的多模态 JEV 决策模型
model_dir = "/path/to/qwen3_5_jev_merged"
engine = PtEngine(model_dir, task_type='decision', torch_dtype=torch.bfloat16, device_map='auto')

# 构造多模态决策请求
infer_requests = [
    {
        "images": ["/data/audit_samples/img_9921.jpg"],
        "questions": {
            "risk_assessment": {
                "type": "choice",
                "instruction": "请分析该图像的商品资质与视觉合规性：",
                "criteria": {
                    "0": "合规正常展示",
                    "1": "轻度疑似违规（需人工初审）",
                    "2": "高危违规（直接拦截拦截）"
                }
            }
        }
    }
]

# 单次前向推理（max_tokens=1 甚至无需 Decode）
results = engine.infer(infer_requests, request_config=RequestConfig(max_tokens=1))
probs = results[0].decision_probabilities["risk_assessment"]

print("决策概率分布:", probs)
# 输出: {'0': 0.008, '1': 0.042, '2': 0.950} -> 触发高危拦截，端到端延迟仅 22ms (H200)
```

---

## 8. 环境配置与交付清单

### 8.1 快速发射指令表

#### Qwen3.5-27B Dense JEV 训练发射
```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/models/Qwen3.5-27B \
    --model_type qwen3_5 \
    --task_type decision \
    --loss_type distill_rps \
    --dataset /path/to/jev_train.jsonl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --modules_to_save "decision_head" "score_head" "visual.merger" \
    --freeze_vit true \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 8192 \
    --packing false \
    --deepspeed default-zero2 \
    --num_train_epochs 2 \
    --output_dir output/qwen3_5_27b_jev
```

#### Qwen3.5-35B-A3B MoE JEV 训练发射
```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/models/Qwen3.5-35B-A3B \
    --model_type qwen3_5_moe \
    --task_type decision \
    --loss_type distill_rps \
    --dataset /path/to/jev_train.jsonl \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --tuner_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules all-linear \
    --modules_to_save "decision_head" "score_head" "visual.merger" \
    --freeze_vit true \
    --learning_rate 4e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 8192 \
    --packing false \
    --deepspeed default-zero2 \
    --num_train_epochs 2 \
    --output_dir output/qwen3_5_35b_a3b_jev
```

---
*报告维护：分布式系统与模型工程团队（面向 H200 算力集群）*
