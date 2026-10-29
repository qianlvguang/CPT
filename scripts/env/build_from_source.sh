#!/usr/bin/env bash
# PyTorch 2.14.1 + CUDA 13.0 源码编译骨架（目标架构 sm90 / sm100 / sm120）
#
# ⚠️ 仅在官方轮子不满足需求时使用（见 README 第三节）。
#    官方 cu130 轮子已原生支持这三个架构，通常**不需要**本脚本。
#
# 运行环境要求：
#   - Linux x86_64，glibc >= 2.28（推荐 Ubuntu 22.04/24.04 或 manylinux_2_28 容器）
#   - CUDA Toolkit 13.0 + cuDNN 9.x + NCCL（与 CUDA 13 匹配）
#   - RAM >= 64GB，磁盘 >= 80GB，CPU 核数越多越好
#   - 预计耗时 2~6 小时
#
# 用法：
#   ./build_from_source.sh                # 完整构建
#   MAX_JOBS=8 ./build_from_source.sh     # 内存较小时限制并行

set -euo pipefail

TORCH_TAG="v2.14.1"
WORK="${WORK:-$PWD/pytorch-build}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0;12.0}"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export USE_CUDA=1
export USE_NCCL="${USE_NCCL:-1}"
export FORCE_CUDA=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export BUILD_TEST=0

echo "=== 配置 ==="
echo "  PyTorch tag     : $TORCH_TAG"
echo "  工作目录        : $WORK"
echo "  TORCH_CUDA_ARCH_LIST: $ARCH_LIST"
echo "  MAX_JOBS        : $MAX_JOBS"
echo "  CUDA_HOME       : $CUDA_HOME"
echo

echo "=== 环境自检 ==="
command -v nvcc >/dev/null || { echo "❌ 找不到 nvcc（CUDA Toolkit 未安装或不在 PATH）"; exit 1; }
nvcc --version | tail -2
echo "  CMake: $(cmake --version 2>/dev/null | head -1 || echo '未安装')"
echo "  Ninja: $(ninja --version 2>/dev/null || echo '未安装')"
echo "  编译内存需求: 建议 >= 64GB（链接阶段最吃内存）"
echo

mkdir -p "$WORK"
cd "$WORK"

if [ ! -d pytorch ]; then
  echo "=== 1/4 克隆源码（含全部子模块）==="
  git clone --recursive --depth 1 --branch "$TORCH_TAG" \
    https://github.com/pytorch/pytorch.git
  cd pytorch
  git submodule sync --recursive
  git submodule update --init --recursive --depth 1
else
  echo "=== 1/4 使用已有源码 ==="
  cd pytorch
fi

echo
echo "=== 2/4 安装 Python 构建依赖 ==="
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements.txt
python3 -m pip install ninja cmake

echo
echo "=== 3/4 开始编译（这一步最久）==="
echo "  目标架构: $TORCH_CUDA_ARCH_LIST"
echo "  提示: 若被 OOM 杀掉，重跑时把 MAX_JOBS 调小"
python3 setup.py bdist_wheel

echo
echo "=== 4/4 完成 ==="
ls -lh dist/*.whl 2>/dev/null || echo "⚠️ dist/ 下没有产物，请检查日志"
echo
echo "安装方式："
echo "  pip install dist/torch-*.whl"
echo
echo "验证方式："
echo "  $(dirname "$0")/verify_arch.sh"
