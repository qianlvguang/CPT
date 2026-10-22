#!/usr/bin/env bash
# ==============================================================================
# Qwen3.5-35B-A3B-QSA 推理引擎极限优化启动脚本 (vLLM & SGLang on H200)
#
# 支持模式:
#   bash serve_vllm_sglang_h200.sh prepare_config  # 校验/补齐 config.json 中的 QSA Indexer 与 YaRN 字段
#   bash serve_vllm_sglang_h200.sh vllm            # 启动 vLLM (H200 TP=2 零 KV 复制 + MTP 投机解码)
#   bash serve_vllm_sglang_h200.sh vllm_fp8_ep8    # 启动 vLLM 8×H200 高并发集群模式 (EP=8 + FP8 + QSA)
#   bash serve_vllm_sglang_h200.sh sglang          # 启动 SGLang (NEXTN 投机解码 + QSA 索引复用 + FlashInfer)
# ==============================================================================
set -euo pipefail

MODE="${1:-sglang}"
WORK_DIR="${WORK_DIR:-/data/autovision-cfs/autovision-notebook}"
MODEL_PATH="${MODEL_PATH:-${WORK_DIR}/algo4/tpz/ft/qwen3_5_35b_a3b_qsa_muon/sft_full}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # 官方建议至少保留 128K 以维持完整 thinking 与 QSA 优势
TEXT_ONLY="${TEXT_ONLY:-0}"                # 设为 1 时追加 --language-model-only 跳过 ViT 省显存

# ⚠️ 核心架构契合点:
# Qwen3.5-35B-A3B 的 num_key_value_heads = 2!
# 单实例低延迟推理用 TP=2 (刚好每卡 1 个 KV 头，零 KV 复制冗余，且单卡 H200 141GB 装 17.5B 绰绰有余)；
# 8 卡满吞吐推理用 DP=4 × TP=2 或 EP=8。
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"

case "${MODE}" in
  prepare_config)
    python3 - "${MODEL_PATH}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
cfg_path = model_dir / "config.json"
if not cfg_path.is_file():
    raise FileNotFoundError(f"未找到配置文件: {cfg_path}")

cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
text_cfg = cfg.setdefault("text_config", {})

# 补齐 Qwen3.8-Flash-Next QSA Indexer 推理必需字段
qsa_defaults = {
    "full_attention_interval": 4,
    "head_dim": 256,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "indexer_head_dim": 128,
    "indexer_kv_heads": 1,
    "indexer_n_heads": 4,
    "mamba_ssm_dtype": "float32",
}
for k, v in qsa_defaults.items():
    text_cfg.setdefault(k, v)

backup = model_dir / "config.json.bak"
if not backup.exists():
    backup.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[prepare_config] 已核验并补齐 QSA Indexer 字段: {cfg_path}")
PY
    ;;

  vllm)
    # --------------------------------------------------------------------------
    # vLLM 推荐配置 (TP=2, 100% 契合 2 个 KV 头 + Paged QSA Side-Cache + MTP)
    # --------------------------------------------------------------------------
    EXTRA_ARGS=()
    if [[ "${TEXT_ONLY}" == "1" ]]; then
      EXTRA_ARGS+=(--language-model-only)
    fi
    if [[ "${ENABLE_MTP:-1}" == "1" ]]; then
      EXTRA_ARGS+=(--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
    fi

    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --enable-expert-parallel \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization 0.90 \
      --mamba-ssm-cache-dtype float32 \
      --enable-chunked-prefill \
      --max-num-batched-tokens 16384 \
      --reasoning-parser qwen3 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      "${EXTRA_ARGS[@]}"
    ;;

  vllm_fp8_ep8)
    # --------------------------------------------------------------------------
    # vLLM 8×H200 极限吞吐模式 (TP=2 + DP=4 或 TP=8 + EP=8 + FP8 动态量化)
    # --------------------------------------------------------------------------
    vllm serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tensor-parallel-size 2 \
      --data-parallel-size 4 \
      --enable-expert-parallel \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --mamba-ssm-cache-dtype float32 \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization 0.92 \
      --enable-chunked-prefill \
      --max-num-batched-tokens 32768 \
      --reasoning-parser qwen3
    ;;

  sglang)
    # --------------------------------------------------------------------------
    # SGLang 推荐配置 (FlashInfer + QSA Triton 内核 + NEXTN 投机解码索引共享)
    # --------------------------------------------------------------------------
    EXTRA_ARGS=()
    if [[ "${TEXT_ONLY}" == "1" ]]; then
      EXTRA_ARGS+=(--language-model-only)
    fi
    if [[ "${ENABLE_MTP:-1}" == "1" ]]; then
      EXTRA_ARGS+=(
        --speculative-algo NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
      )
    fi

    python3 -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --tp-size "${TP_SIZE}" \
      --ep-size "${EP_SIZE}" \
      --context-length "${MAX_MODEL_LEN}" \
      --mem-fraction-static 0.88 \
      --chunked-prefill-size 16384 \
      --attention-backend flashinfer \
      --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder \
      "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "未知模式: ${MODE} (可选: prepare_config | vllm | vllm_fp8_ep8 | sglang)" >&2
    exit 1
    ;;
esac
