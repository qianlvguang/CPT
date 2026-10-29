#!/usr/bin/env bash
# 安装官方 PyTorch 2.14.1 + CUDA 13.0 轮子（已内置 sm_75/80/86/90/100/120）
#
# 用法：
#   ./install_official.sh              # 只装 torch
#   ./install_official.sh --with-vision # 同时装 torchvision
#   ./install_official.sh --dry-run    # 只打印将执行的命令
#
# 前置：NVIDIA 驱动 >= 580.x（CUDA 13.0 要求）

set -euo pipefail

INDEX="https://download.pytorch.org/whl/cu130"
TORCH_VER="2.14.1"
VISION_VER="0.29.1"
AUDIO_VER="2.11.0"     # 注意：cu130 下 torchaudio 最新为 2.11.0，与 torch 2.14 的配对需自行确认

WITH_VISION=0
DRY_RUN=0
WITH_AUDIO=0
for arg in "$@"; do
  case "$arg" in
    --with-vision) WITH_VISION=1 ;;
    --with-audio)  WITH_AUDIO=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    -h|--help)     sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg" >&2; exit 1 ;;
  esac
done

PY="${PYTHON:-python3}"
echo "=== 环境检查 ==="
echo "  Python: $($PY -V 2>&1)"
echo "  平台:   $(uname -s) $(uname -m)"

# glibc 检查（manylinux_2_28 需要 >= 2.28）
if ldd --version 2>/dev/null | head -1 | grep -q glibc; then
  GLIBC=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$')
  echo "  glibc:  ${GLIBC}  （需要 >= 2.28）"
fi

# 驱动检查
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  echo "  驱动:   ${DRV}"
  echo "  显卡:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
else
  echo "  ⚠️ 未找到 nvidia-smi —— 本机可能没有 NVIDIA 驱动/GPU"
fi
echo

CMD=("$PY" -m pip install
  "torch==${TORCH_VER}"
)
if [ "$WITH_VISION" -eq 1 ]; then CMD+=("torchvision==${VISION_VER}"); fi
if [ "$WITH_AUDIO" -eq 1 ]; then CMD+=("torchaudio==${AUDIO_VER}"); fi
CMD+=("--index-url" "$INDEX")

echo "=== 将执行 ==="
printf '  %q ' "${CMD[@]}"; echo
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "（--dry-run：未实际执行）"
  exit 0
fi

"${CMD[@]}"

echo
echo "=== 安装完成，开始验证 ==="
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA: ", torch.version.cuda)
print("arch: ", torch.cuda.get_arch_list())
print("可用: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("当前卡:", torch.cuda.get_device_name(0))
    print("算力:  ", torch.cuda.get_device_capability(0))
required = {"sm_90", "sm_100", "sm_120"}
have = set(torch.cuda.get_arch_list())
missing = required - have
if missing:
    print("⚠️ 缺少架构:", missing)
else:
    print("✅ sm_90 / sm_100 / sm_120 全部支持")
PY
