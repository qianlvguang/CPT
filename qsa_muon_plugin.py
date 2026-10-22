"""Qwen3.5-35B-A3B (QSA 替换 FullAttn + Muon 优化器) 专用 ms-swift / Megatron 插件。

核心职责：
1. 分阶段训练控制 (QSA_TRAIN_STAGE):
   - warmup   : 仅放开 QSA Indexer (或 10 个 QSA 注意力层)，对齐稀疏路由，冻结 MoE 与 GDN 主干。
   - cpt      : 放开 LLM 主干 + QSA + MoE 专家进行全参 CPT 能力重建（默认冻结 ViT，可通过环境变量放开）。
   - sft_full : 全参 SFT（冻结 ViT 与 MoE Router，防止路由坍缩）。
   - sft_lora : 异构 LoRA SFT（覆盖 QSA + GDN + Shared Expert，可选冻结视觉 LoRA）。
2. Muon 参数分组安全护栏:
   - 严禁对 Embedding / LM Head / 1D Norm / GDN (A_log, dt_bias, conv1d) / MoE Router / QSA Indexer 做 Newton-Schulz 正交化，
     确保这些模块走标量/AdamW 分支，仅对内部 2D 线性矩阵应用 Muon。
3. 严格数据预处理护栏:
   - 继承 recommended_pipeline_plugin 的不静默丢行与分通道 loss (V1~V4) 支持。
"""
from __future__ import annotations

import os
import re
from typing import Iterable

# ---------------------------------------------------------------------------
# 1. 严格预处理与分通道 Loss 补丁 (兼容 CPT 纯文本/多模态预训练与 SFT)
# ---------------------------------------------------------------------------
try:
    from swift.dataset.preprocessor.core import RowPreprocessor

    _original_preprocess = RowPreprocessor.batched_preprocess

    def _strict_preprocess(self, batched_row, *, strict, ignore_max_length_error):
        row = dict(batched_row)
        if "channel" not in row and "variant" in row:
            row["channel"] = [f"V{int(v)}" if v is not None else None for v in row["variant"]]
        enforce_strict = os.environ.get("QSA_STRICT_DATA", "1") == "1"
        result = _original_preprocess(
            self,
            row,
            strict=True if enforce_strict else strict,
            ignore_max_length_error=False if enforce_strict else ignore_max_length_error,
        )
        if enforce_strict and "variant" in row and len(result.get("messages", [])) != len(row["variant"]):
            raise ValueError("[qsa_muon_plugin] 检测到预处理丢行；请检查 max_length 或异常超长样本。")
        return result

    if not getattr(RowPreprocessor.batched_preprocess, "_qsa_muon_patched", False):
        _strict_preprocess._qsa_muon_patched = True
        RowPreprocessor.batched_preprocess = _strict_preprocess
        print("[qsa_muon_plugin] RowPreprocessor strict/channel patch installed", flush=True)
except Exception as exc:
    print(f"[qsa_muon_plugin] RowPreprocessor patch skipped: {exc}", flush=True)


# ---------------------------------------------------------------------------
# 2. QSA + Muon 参数判定规则与分阶段冻结控制
# ---------------------------------------------------------------------------
# 必须排除在 Muon Newton-Schulz 正交化之外的参数名正则（走内部 AdamW）
ADAMW_ONLY_PATTERNS = (
    r"(^|\.)embed_tokens\.",
    r"(^|\.)word_embeddings\.",
    r"(^|\.)lm_head\.",
    r"(^|\.)output_layer\.",
    r"(^|\.)visual\.",
    r"(^|\.)indexer\.",          # QSA 轻量索引器 (index_qk_proj / q_layernorm / k_layernorm)
    r"(^|\.)mlp\.gate\.",        # MoE 路由器 (256 专家路由对正交化极敏感)
    r"(^|\.)router\.",           # Megatron MoE router 命名
    r"(^|\.)shared_expert_gate\.",
    r"(^|\.)A_log$",             # Gated DeltaNet 递归衰减参数
    r"(^|\.)dt_bias$",           # Gated DeltaNet 时间步偏置
    r"(^|\.)conv1d\.",           # Gated DeltaNet 短卷积核
    r"norm",                     # 各类 RMSNorm / LayerNorm (q_norm, k_norm 等)
    r"bias$",
)
_ADAMW_REGEX = re.compile("|".join(ADAMW_ONLY_PATTERNS))


def is_adamw_only_param(name: str, param) -> bool:
    """判断参数是否必须回退到 AdamW（禁止使用 Muon 正交更新）。"""
    if param.ndim != 2:
        return True
    if _ADAMW_REGEX.search(name):
        return True
    return False


def configure_trainable_and_muon_flags(models: Iterable) -> dict:
    """根据 QSA_TRAIN_STAGE 设置 requires_grad 并为参数打上 Muon/AdamW 分组标记。"""
    stage = os.environ.get("QSA_TRAIN_STAGE", "cpt").strip().lower()
    # warmup 阶段可选只训 indexer ("indexer") 或训完整 QSA 层 ("qsa_layer")
    warmup_scope = os.environ.get("QSA_WARMUP_SCOPE", "qsa_layer").strip().lower()
    freeze_vit = os.environ.get("QSA_FREEZE_VIT", "1") == "1"
    freeze_router = os.environ.get("QSA_FREEZE_ROUTER", "0") == "1"

    stats = {
        "stage": stage,
        "total_params": 0,
        "trainable_params": 0,
        "muon_2d_params": 0,
        "adamw_fallback_params": 0,
        "indexer_params": 0,
        "router_params": 0,
    }

    for model in models:
        for name, param in model.named_parameters():
            numel = param.numel()
            stats["total_params"] += numel

            is_visual = name.startswith("visual.") or ".visual." in name
            is_indexer = ".indexer." in name or "index_qk_proj" in name
            is_qsa_attn = is_indexer or (
                (".self_attn." in name or ".self_attention." in name)
                and not ("linear_attn" in name or "deltanet" in name)
            )
            is_router = ".mlp.gate." in name or ".router." in name
            is_lora = ".lora_A." in name or ".lora_B." in name

            if is_indexer:
                stats["indexer_params"] += numel
            if is_router:
                stats["router_params"] += numel

            # 1) 决定 requires_grad
            if stage == "warmup":
                # Stage 0: 仅对齐 QSA Indexer 或 QSA 全注意力层，冻结 MoE 与 GDN
                if warmup_scope == "indexer":
                    trainable = is_indexer
                else:
                    trainable = is_qsa_attn and not is_visual
            elif stage in ("cpt", "sft_full"):
                # Stage 1 / 2A: 全参 CPT 或全参 SFT
                trainable = True
                if freeze_vit and is_visual:
                    trainable = False
                if freeze_router and is_router:
                    trainable = False
            elif stage == "sft_lora":
                # Stage 2B: LoRA SFT
                trainable = is_lora and (not freeze_vit or not is_visual)
            else:
                raise ValueError(f"[qsa_muon_plugin] 未知的 QSA_TRAIN_STAGE={stage}")

            param.requires_grad_(trainable)

            if trainable:
                stats["trainable_params"] += numel
                if is_adamw_only_param(name, param):
                    # 为 ms-swift / Megatron Muon 优化器标记非 Muon 参数
                    setattr(param, "use_muon", False)
                    stats["adamw_fallback_params"] += numel
                else:
                    setattr(param, "use_muon", True)
                    stats["muon_2d_params"] += numel

    if stats["trainable_params"] == 0:
        raise RuntimeError(f"[qsa_muon_plugin] stage={stage} 可训练参数为 0！请检查模块命名或配置。")

    print(
        f"[qsa_muon_plugin] stage={stage} | "
        f"trainable={stats['trainable_params']:,} / {stats['total_params']:,} | "
        f"muon_2d={stats['muon_2d_params']:,} | "
        f"adamw_fallback={stats['adamw_fallback_params']:,} | "
        f"indexer={stats['indexer_params']:,} | router={stats['router_params']:,}",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# 3. 挂载到 Megatron-SWIFT 与 HF Trainer 生命周期 (确保在 optimizer 创建前执行)
# ---------------------------------------------------------------------------
try:
    from swift.megatron.trainers.base import BaseMegatronTrainer

    if hasattr(BaseMegatronTrainer, "get_optimizer_and_scheduler"):
        _orig_get_opt = BaseMegatronTrainer.get_optimizer_and_scheduler

        def _patched_get_opt(self, models, *args, **kwargs):
            model_list = models if isinstance(models, (list, tuple)) else [models]
            configure_trainable_and_muon_flags(model_list)
            return _orig_get_opt(self, models, *args, **kwargs)

        if not getattr(BaseMegatronTrainer.get_optimizer_and_scheduler, "_qsa_muon_patched", False):
            _patched_get_opt._qsa_muon_patched = True
            BaseMegatronTrainer.get_optimizer_and_scheduler = _patched_get_opt
            print("[qsa_muon_plugin] BaseMegatronTrainer.get_optimizer_and_scheduler hook installed", flush=True)
    elif hasattr(BaseMegatronTrainer, "_prepare_peft_model"):
        _orig_prepare_peft = BaseMegatronTrainer._prepare_peft_model

        def _patched_prepare_peft(self, models):
            peft_models = _orig_prepare_peft(self, models)
            configure_trainable_and_muon_flags(models)
            return peft_models

        if not getattr(BaseMegatronTrainer._prepare_peft_model, "_qsa_muon_patched", False):
            _patched_prepare_peft._qsa_muon_patched = True
            BaseMegatronTrainer._prepare_peft_model = _patched_prepare_peft
            print("[qsa_muon_plugin] BaseMegatronTrainer._prepare_peft_model hook installed", flush=True)
except Exception as exc:
    print(f"[qsa_muon_plugin] Megatron trainer hook skipped (non-Megatron mode?): {exc}", flush=True)

try:
    from swift.trainers import SwiftMixin

    _orig_create_optimizer = SwiftMixin.create_optimizer

    def _patched_create_optimizer(self, *args, **kwargs):
        if getattr(self, "model", None) is not None:
            configure_trainable_and_muon_flags([self.model])
        return _orig_create_optimizer(self, *args, **kwargs)

    if not getattr(SwiftMixin.create_optimizer, "_qsa_muon_patched", False):
        _patched_create_optimizer._qsa_muon_patched = True
        SwiftMixin.create_optimizer = _patched_create_optimizer
        print("[qsa_muon_plugin] SwiftMixin.create_optimizer hook installed", flush=True)
except Exception as exc:
    print(f"[qsa_muon_plugin] SwiftMixin hook skipped: {exc}", flush=True)
