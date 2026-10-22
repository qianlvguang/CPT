#!/usr/bin/env bash
# ==============================================================================
# Qwen3.5-35B-A3B (QSA 替换 FullAttn + Muon 优化器) CPT 重建与 SFT 训练脚本
# 硬件目标: NVIDIA H200 (141GB HBM3e) × 8 (NVLink 900GB/s)
#
# 用法:
#   bash run_cpt_sft_qsa_35a3_h200x8.sh warmup     # 阶段0: 仅预热对齐 QSA Indexer / QSA 层
#   bash run_cpt_sft_qsa_35a3_h200x8.sh cpt        # 阶段1: Muon 全参 CPT 重建 35B-A3B 能力与长上下文
#   bash run_cpt_sft_qsa_35a3_h200x8.sh sft_full   # 阶段2A: Muon 全参 SFT (冻结 ViT 与 MoE Router)
#   bash run_cpt_sft_qsa_35a3_h200x8.sh sft_lora   # 阶段2B: 异构 LoRA SFT (QSA + GDN + Shared Expert)
#   SMOKE=1 bash run_cpt_sft_qsa_35a3_h200x8.sh cpt # 10 步冒烟测试 (验证显存/吞吐/算子挂载)
#   BACKEND=hf bash run_cpt_sft_qsa_35a3_h200x8.sh cpt # 切换至 HF + DeepSpeed ZeRO-2 后端
# ==============================================================================
set -euo pipefail

STAGE="${1:-cpt}"
BACKEND="${BACKEND:-megatron}"   # megatron (推荐，高 MFU) | hf (自定义 HF modeling 零移植直接跑)
SMOKE="${SMOKE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_PATH="${SCRIPT_DIR}/qsa_muon_plugin.py"

# ------------------------------------------------------------------------------
# 1. 路径配置 (支持环境变量覆盖，不写死本地绝对路径)
# ------------------------------------------------------------------------------
WORK_DIR="${WORK_DIR:-/data/autovision-cfs/autovision-notebook}"
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/algo4/models/Qwen3.5-35B-A3B-QSA}"
CPT_DATA_DIR="${CPT_DATA_DIR:-${WORK_DIR}/algo4/tpz/cpt_data}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${WORK_DIR}/algo4/tpz/test}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORK_DIR}/algo4/tpz/ft/qwen3_5_35b_a3b_qsa_muon}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${SCRIPT_DIR}/ds_zero2_h200.json}"

if [[ "${SMOKE}" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/${STAGE}_smoke"
else
  OUTPUT_DIR="${OUTPUT_ROOT}/${STAGE}"
fi
mkdir -p "${OUTPUT_DIR}"

# ------------------------------------------------------------------------------
# 2. H200 × 8 硬件级与通信级满血加速环境变量
# ------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS="1"          # Megatron 序列并行与通信重叠必需
export NCCL_NVLS_ENABLE="1"                     # 启用 H200 NVLink SHARP 集合通信卸载
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"      # 减少流同步显存碎片
export TOKENIZERS_PARALLELISM="False"
export OMP_NUM_THREADS="4"
export MODELSCOPE_CACHE_HOME="${MODELSCOPE_CACHE_HOME:-${WORK_DIR}/logos-remote/sft/cache/${STAGE}}"

# 视觉 Token 预算控制
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-16384}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"

# ------------------------------------------------------------------------------
# 3. 并行拓扑设计 (针对 35B-A3B 的关键硬件约束)
#    ⚠️ 35B-A3B 的 num_key_value_heads = 2, QSA indexer_kv_heads = 1!
#    若盲目套用 27B 的 TP=4 会触发 KV 头整除报错或被迫复制 KV 头浪费通信。
#    因此:
#      - 默认推荐 TP=1, EP=8, DP=8 (或长序列 32K 时 TP=2, EP=4, DP=4, 需开启复制或仅切 Q/FFN)
# ------------------------------------------------------------------------------
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
CP_SIZE="${CP_SIZE:-1}"
SP_ENABLE="${SP_ENABLE:-false}"
if [[ "${TP_SIZE}" -gt 1 ]]; then
  SP_ENABLE="true"
fi

# ------------------------------------------------------------------------------
# 4. 分阶段超参数与训练策略配置
# ------------------------------------------------------------------------------
export QSA_TRAIN_STAGE="${STAGE}"
export QSA_WARMUP_SCOPE="${QSA_WARMUP_SCOPE:-qsa_layer}" # indexer | qsa_layer

case "${STAGE}" in
  warmup)
    # 阶段 0: 仅训练 QSA Indexer / QSA 注意力层，快速对齐微块稀疏路由 (约 1B~2B tokens)
    TASK_CMD="pt"
    OPTIMIZER_NAME="adam"
    TUNER_TYPE="full"
    MAX_LENGTH="${MAX_LENGTH:-16384}"
    MBS="${MBS:-2}"
    GBS="${GBS:-64}"
    LR="${LR:-5e-4}"
    MIN_LR="${MIN_LR:-5e-5}"
    WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    EPOCHS="${EPOCHS:-1}"
    export QSA_FREEZE_VIT="1"
    export QSA_FREEZE_ROUTER="1"
    DATASETS=("${CPT_DATA_DIR}/cpt_qsa_warmup_16k.parquet")
    ;;
  cpt)
    # 阶段 1: Muon 全参 CPT 重建 35B-A3B 能力 (AdamW -> Muon 切换需 5%~10% 较长 warmup)
    TASK_CMD="pt"
    OPTIMIZER_NAME="dist_muon"
    TUNER_TYPE="full"
    MAX_LENGTH="${MAX_LENGTH:-16384}"
    MBS="${MBS:-2}"
    GBS="${GBS:-64}"
    LR="${LR:-1.5e-4}"          # Muon 2D 矩阵主学习率 (配合 spectral / rms 缩放)
    MIN_LR="${MIN_LR:-1.5e-5}"
    WARMUP_FRAC="${WARMUP_FRAC:-0.08}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
    EPOCHS="${EPOCHS:-1}"
    export QSA_FREEZE_VIT="${QSA_FREEZE_VIT:-1}"
    export QSA_FREEZE_ROUTER="${QSA_FREEZE_ROUTER:-0}"
    DATASETS=(
      "${CPT_DATA_DIR}/cpt_general_long_context.parquet"
      "${CPT_DATA_DIR}/cpt_domain_multimodal_replay.parquet"
    )
    ;;
  sft_full)
    # 阶段 2A: Muon 全参 SFT (默认冻结 ViT 与 MoE Router 防止微调阶段路由坍缩)
    TASK_CMD="sft"
    OPTIMIZER_NAME="dist_muon"
    TUNER_TYPE="full"
    MAX_LENGTH="${MAX_LENGTH:-8192}"
    MBS="${MBS:-2}"
    GBS="${GBS:-32}"
    LR="${LR:-2e-5}"
    MIN_LR="${MIN_LR:-2e-6}"
    WARMUP_FRAC="${WARMUP_FRAC:-0.03}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
    EPOCHS="${EPOCHS:-2}"
    export QSA_FREEZE_VIT="${QSA_FREEZE_VIT:-1}"
    export QSA_FREEZE_ROUTER="${QSA_FREEZE_ROUTER:-1}"
    DATASETS=(
      "${SFT_DATA_DIR}/train_caption_ultra_x_v1mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v2mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v3mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v4mini.parquet"
    )
    ;;
  sft_lora)
    # 阶段 2B: 异构 LoRA SFT (省显存/防遗忘，挂 QSA + GDN + Shared Expert)
    TASK_CMD="sft"
    OPTIMIZER_NAME="${OPTIMIZER_NAME:-adam}"
    TUNER_TYPE="lora"
    MAX_LENGTH="${MAX_LENGTH:-8192}"
    MBS="${MBS:-4}"
    GBS="${GBS:-32}"
    LR="${LR:-3e-5}"
    MIN_LR="${MIN_LR:-3e-6}"
    WARMUP_FRAC="${WARMUP_FRAC:-0.03}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    EPOCHS="${EPOCHS:-2}"
    export QSA_FREEZE_VIT="${QSA_FREEZE_VIT:-0}"
    export QSA_FREEZE_ROUTER="1"
    DATASETS=(
      "${SFT_DATA_DIR}/train_caption_ultra_x_v1mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v2mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v3mini.parquet"
      "${SFT_DATA_DIR}/train_caption_ultra_x_v4mini.parquet"
    )
    ;;
  *)
    echo "未知阶段: ${STAGE} (可选: warmup | cpt | sft_full | sft_lora)" >&2
    exit 1
    ;;
esac

# Smoke 模式仅跑 10 iterations，且必须省略 --num_train_epochs 避免覆盖 train_iters
ITER_ARGS=()
if [[ "${SMOKE}" == "1" ]]; then
  ITER_ARGS=(--train_iters 10)
  SAVE_STEPS=1000000
else
  ITER_ARGS=(--num_train_epochs "${EPOCHS}")
  SAVE_STEPS="${SAVE_STEPS:-500}"
fi

# ------------------------------------------------------------------------------
# 5. 组装并执行训练命令
# ------------------------------------------------------------------------------
if [[ "${BACKEND}" == "megatron" ]]; then
  CMD=(
    megatron "${TASK_CMD}"
    --model "${MODEL_PATH}"
    --dataset "${DATASETS[@]}"
    --split_dataset_ratio 0
    --load_from_cache_file true
    --save_safetensors true
    --strict true
    --truncation_strategy delete
    --dataset_shuffle true
    --data_seed 42
    --tensor_model_parallel_size "${TP_SIZE}"
    --pipeline_model_parallel_size "${PP_SIZE}"
    --context_parallel_size "${CP_SIZE}"
    --expert_model_parallel_size "${EP_SIZE}"
    --expert_tensor_parallel_size 1
    --sequence_parallel "${SP_ENABLE}"
    --use_distributed_optimizer true
    --moe_grouped_gemm true
    --moe_shared_expert_overlap true
    --moe_aux_loss_coeff 0.001
    --recompute_granularity selective
    --attention_backend flash
    --packing true
    --padding_free true
    --cross_entropy_loss_fusion true
    --optimizer "${OPTIMIZER_NAME}"
    --tuner_type "${TUNER_TYPE}"
    --freeze_llm false
    --freeze_vit false
    --freeze_aligner false
    --external_plugins "${PLUGIN_PATH}"
    --enable_channel_loss true
    --micro_batch_size "${MBS}"
    --global_batch_size "${GBS}"
    --max_length "${MAX_LENGTH}"
    --lr "${LR}"
    --min_lr "${MIN_LR}"
    --lr_warmup_fraction "${WARMUP_FRAC}"
    --lr_decay_style cosine
    --weight_decay "${WEIGHT_DECAY}"
    --clip_grad 1.0
    --seed 42
    --finetune true
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 1000000
    --add_version false
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers 16
    --dataset_num_proc 16
    --packing_num_proc 16
    "${ITER_ARGS[@]}"
  )

  # Muon 专属超参 (megatron-core >= 0.16 / 0.18 支持)
  if [[ "${OPTIMIZER_NAME}" == *"muon"* ]]; then
    CMD+=(
      --muon_scale_mode "${MUON_SCALE_MODE:-spectral}"
      --muon_num_ns_steps "${MUON_NUM_NS_STEPS:-5}"
      --muon_fp32_matmul_prec "${MUON_MATMUL_PREC:-high}"
    )
  fi

  # 异构 LoRA 专属参数
  if [[ "${TUNER_TYPE}" == "lora" ]]; then
    CMD+=(
      --target_modules "all-linear"
      --lora_rank 64
      --lora_alpha 128
      --lora_dropout 0.05
      --merge_lora false
    )
  fi

elif [[ "${BACKEND}" == "hf" ]]; then
  # HF + DeepSpeed ZeRO-2 路线 (适用于直接修改了 transformers modeling_qwen3_5_moe.py 的场景)
  GRAD_ACCUM=$(( GBS / (NPROC_PER_NODE * MBS) ))
  if [[ "${GRAD_ACCUM}" -lt 1 ]]; then
    GRAD_ACCUM=1
  fi
  CMD=(
    swift "${TASK_CMD}"
    --model "${MODEL_PATH}"
    --dataset "${DATASETS[@]}"
    --split_dataset_ratio 0
    --load_from_cache_file true
    --strict true
    --truncation_strategy delete
    --packing true
    --padding_free true
    --attn_impl flash_attn
    --torch_dtype bfloat16
    --deepspeed "${DS_CONFIG_PATH}"
    --tuner_type "${TUNER_TYPE}"
    --optim "${OPTIMIZER_NAME}"
    --external_plugins "${PLUGIN_PATH}"
    --enable_channel_loss true
    --per_device_train_batch_size "${MBS}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --max_length "${MAX_LENGTH}"
    --learning_rate "${LR}"
    --warmup_ratio "${WARMUP_FRAC}"
    --lr_scheduler_type cosine
    --weight_decay "${WEIGHT_DECAY}"
    --max_grad_norm 1.0
    --gradient_checkpointing true
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --save_total_limit 1000000
    --add_version false
    --output_dir "${OUTPUT_DIR}"
    --dataloader_num_workers 8
    --dataset_num_proc 16
  )
  if [[ "${SMOKE}" == "1" ]]; then
    CMD+=(--max_steps 10)
  else
    CMD+=(--num_train_epochs "${EPOCHS}")
  fi
else
  echo "未知 BACKEND: ${BACKEND} (可选: megatron | hf)" >&2
  exit 1
fi

echo "=================================================================="
echo "[Launch] Stage=${STAGE} | Backend=${BACKEND} | Optimizer=${OPTIMIZER_NAME}"
echo "[Launch] Model=${MODEL_PATH}"
echo "[Launch] Output=${OUTPUT_DIR}"
echo "[Launch] Command: ${CMD[*]}"
echo "=================================================================="

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
