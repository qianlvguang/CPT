#!/usr/bin/env bash
# 验证已安装的 PyTorch 是否支持 sm90 / sm100 / sm120
#
# 用法： ./verify_arch.sh
# 如果传 wheel 路径，则离线检查该 .whl 内含的架构（不需要安装）：
#        ./verify_arch.sh torch-2.14.1+cu130-cp312-*.whl

set -euo pipefail
PY="${PYTHON:-python3}"

if [ "${1:-}" != "" ] && [ -f "${1:-}" ]; then
  WHL="$1"
  echo "=== 离线检查 wheel: $(basename "$WHL") ==="
  echo "  大小: $(du -h "$WHL" | cut -f1)"
  echo
  echo "  1) 从 libtorch_python.so 读取编译期架构列表："
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  if unzip -o -j "$WHL" "torch/lib/libtorch_python.so" -d "$TMP" >/dev/null 2>&1; then
    ARCHS=$(strings -a "$TMP/libtorch_python.so" 2>/dev/null \
            | grep -E "^sm_[0-9]+( sm_[0-9]+)+$" | head -1)
    echo "     ${ARCHS:-(未找到)}"
    echo
    echo "  2) 检查是否包含目标架构："
    for a in sm_90 sm_100 sm_120; do
      if echo "$ARCHS" | grep -qw "$a"; then echo "     ✅ $a"; else echo "     ❌ $a"; fi
    done
  else
    echo "     ⚠️ 该 wheel 内没有 libtorch_python.so（可能不是 torch 包）"
  fi
  exit 0
fi

echo "=== 已安装环境验证 ==="
"$PY" - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("❌ 未安装 torch")
    sys.exit(1)

print("torch 版本:", torch.__version__)
print("CUDA 版本 :", torch.version.cuda)
archs = torch.cuda.get_arch_list()
print("支持架构  :", archs)
print("CUDA 可用 :", torch.cuda.is_available())

need = {"sm_90", "sm_100", "sm_120"}
have = set(archs)
missing = need - have
print()
for a in sorted(need):
    print(("✅" if a in have else "❌"), a)

print()
if missing:
    print("结论: 缺少 %s —— 官方轮子不该缺，请检查是否装成了 CPU 版或别的 CUDA 版" % sorted(missing))
else:
    print("结论: sm_90 / sm_100 / sm_120 全部支持 ✅")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print("GPU %d: %s  sm_%d%d" % (i, torch.cuda.get_device_name(i), cap[0], cap[1]))
PY
