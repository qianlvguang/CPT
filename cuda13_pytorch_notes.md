# CUDA 13 / PyTorch 2.14.1 环境说明（含 sm90 / sm100 / sm120）

> 最后更新：2026-10-23
> 适用：本仓库 H200×8 训练与推理环境；同样适用于 B200（sm100）、RTX 50 系（sm120）混合集群。

---

## 1. 结论

| 问题 | 结论 |
|---|---|
| 需要自己编译 PyTorch 吗？ | **不需要**。官方 `cu130` 轮子已内置 sm_75/80/86/90/100/120 |
| 线上是 CUDA 13.1 怎么办？ | **装 `cu130` 轮子**（官方无 `cu131` 索引） |
| H200（sm_90）能直接用吗？ | 能，官方轮子含 sm_90 的 SASS + PTX |

---

## 2. 证据（实测官方轮子）

对 `torch-2.14.1+cu130-cp312-manylinux_2_28_x86_64.whl`（554 MB）解包检查：

| 检查项 | 结果 |
|---|---|
| 编译期架构列表（`torch.cuda.get_arch_list()` 来源） | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` |
| `libtorch_cuda.so` 中 SASS 出现次数 | sm_90 → 277 次 · sm_100 → 122 次 · sm_120 → 121 次 |
| 内嵌 PTX（前向兼容） | `compute_75 / 80 / 86 / 90 / 100 / 120` |

> `get_arch_list()` 只返回 SASS 列表，不含 `compute_*` 是正常的；PTX 用于新架构的 JIT。

---

## 3. 安装

```bash
# 现状基线（H200 / sm_90）
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# 需要 torchvision（0.29.x 对应 torch 2.14）
pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

**前置要求**：NVIDIA 驱动 ≥ 580.x（CUDA 13.0 要求）、glibc ≥ 2.28、Python 3.10~3.15。

---

## 4. 线上是 CUDA 13.1 时的选型

PyTorch 官方**只发布三个 CUDA 13 变体**（索引探测）：

| 索引 | HTTP | 有 torch 2.14.1 |
|---|---|---|
| `cu130` | 200 | ✅ |
| `cu131` | **403（不存在）** | — |
| `cu132` | 200 | ✅ |
| `cu133` | 403 | — |
| `cu134` | 200 | ✅ |

依据 **CUDA 次版本兼容（minor version compatibility）**：

> 较旧次版本编译的二进制可在较新次版本的驱动/运行时上运行；反之不行。

- `cu130` → 跑在 **13.1** 上 ✅
- `cu132` / `cu134` → 需要 13.2 / 13.4 驱动，**13.1 不支持** ❌

**验证命令**：

```bash
nvidia-smi | grep "CUDA Version"    # 驱动能力上限
nvcc --version                       # 本机 Toolkit
python -c "import torch; print(torch.version.cuda)"   # 轮子运行时
```

判断规则：`torch.version.cuda (13.0)` ≤ `nvidia-smi` 的 CUDA Version (13.1) → 满足。

---

## 5. 安装后验证

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)          # 2.14.1+cu130
print("CUDA :", torch.version.cuda)         # 13.0
print("arch :", torch.cuda.get_arch_list()) # ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
print("可用 :", torch.cuda.is_available())
print("卡   :", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
PY
```

H200 期望输出 `sm_90`（`(9, 0)`）；B200 为 `sm_100`；RTX 50 系为 `sm_120`。

> 装完务必跑一次 `SMOKE=1 bash run_cpt_sft_qsa_35a3_h200x8.sh warmup` 冒烟，
> 确认 flash-attn / transformer_engine / grouped_gemm 等算子包与 CUDA 13 ABI 对齐。

---

## 6. 什么情况才需要自编译

1. 需要轮子未覆盖的架构（如 sm_103 / sm_110 / sm_121）
2. 想裁剪体积（554 MB → 约 250 MB，仅保留所需架构）
3. 要非官方的 CUDA 组合（13.2/13.4 之外）
4. 要自定义 `USE_*` 开关

编译要点（开销：2~6 小时，RAM 64 GB+，磁盘 80 GB+）：

```bash
git clone --recursive --depth 1 --branch v2.14.1 https://github.com/pytorch/pytorch.git
cd pytorch
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"   # 按需裁剪
export MAX_JOBS=$(nproc)                        # 内存不足时调小
python setup.py bdist_wheel
```

推荐在官方 manylinux_2_28 容器内构建，避免 glibc/编译器差异。

---

## 7. 常见坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 驱动过旧 | `CUDA driver version is insufficient` | 升级到 ≥ 580.x |
| 装了 cu132/cu134 | 13.1 机器上初始化失败 | 换回 cu130 |
| 算子包 ABI 不匹配 | `undefined symbol` / 找不到 cu13 轮子 | 用 cu13 版本或源码编译；必要时 `CUDA_HOME` 指向系统 CUDA 并优先系统库 |
| 内存不足 | 编译时被 OOM Killer 杀 | 降低 `MAX_JOBS`（如 8） |
| 误装 CPU 版 | `torch.cuda.is_available() == False` | 确认用了 `--index-url .../cu130` |
| 子模块缺失 | 编译缺头文件 | `git submodule update --init --recursive` |
