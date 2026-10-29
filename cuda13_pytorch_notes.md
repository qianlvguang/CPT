# PyTorch 2.14.1 + CUDA 13.0 + sm90/sm100/sm120 —— 完整资料

> 整理日期：2026-10-23
> 结论先行：**不需要自己编译。官方 cu130 轮子已经内置了 sm_90 / sm_100 / sm_120，且附带 PTX。**
> 配套脚本位于 [`scripts/env/`](scripts/env/)。

---

## 一、结论与证据

### 官方轮子存在

索引地址：<https://download.pytorch.org/whl/cu130/>

```
torch-2.14.1+cu130-cp310-cp310-manylinux_2_28_x86_64.whl
torch-2.14.1+cu130-cp311-cp311-manylinux_2_28_x86_64.whl
torch-2.14.1+cu130-cp312-cp312-manylinux_2_28_x86_64.whl
torch-2.14.1+cu130-cp313-cp313-manylinux_2_28_x86_64.whl
torch-2.14.1+cu130-cp314-cp314-manylinux_2_28_x86_64.whl
torch-2.14.1+cu130-cp315-cp315-manylinux_2_28_x86_64.whl   （另有 aarch64 / win_amd64）
```

体积约 **554 MB**。

### 架构支持（实测）

从官方 `torch-2.14.1+cu130-cp312-manylinux_2_28_x86_64.whl` 中提取 `torch/lib/libtorch_python.so` 检查：

| 检查项 | 结果 |
|---|---|
| 编译期架构列表（`torch.cuda.get_arch_list()` 的来源） | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` |
| `libtorch_cuda.so` 中 SASS 出现次数 | `sm_90` 277 次 · `sm_100` 122 次 · `sm_120` 121 次 |
| 内嵌 PTX（前向兼容） | `compute_75 / 80 / 86 / 90 / 100 / 120` |

**⇒ sm90（H100/H200）、sm100（B100/B200）、sm120（RTX 50 系）全部原生支持，无需编译。**

> 注：`get_arch_list()` 不包含 `compute_*` 是正常的 —— 它只返回 SASS 列表；PTX 用于未来新架构的 JIT。

---

## 二、直接安装（推荐）

### 命令

```bash
# Python 3.12 / Linux x86_64 —— 按你的实际版本改 cp 标签
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# 需要 torchvision（0.29.x 与 torch 2.14 对应）
pip install torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

或使用附带的脚本：

```bash
./install_official.sh            # 默认 Python 3.12
./install_official.sh cp313      # 指定 Python 标签（仅用于打印说明）
```

### 前置要求

| 项目 | 要求 |
|---|---|
| NVIDIA 驱动 | **≥ 580.x**（CUDA 13.0 的最低驱动要求，以 NVIDIA 官方为准） |
| GPU 架构 | sm_75 及以上（Turing / Ampere / Ada / Hopper / Blackwell） |
| Python | 3.10 ~ 3.15 |
| glibc | ≥ 2.28（wheel 为 `manylinux_2_28`） |
| 平台 | Linux x86_64 / Linux aarch64 / Windows x86_64 |

### 安装后验证

```bash
./verify_arch.sh
```

或手动：

```python
import torch
print(torch.__version__)          # 2.14.1+cu130
print(torch.version.cuda)         # 13.0
print(torch.cuda.get_arch_list()) # ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']
print(torch.cuda.is_available())
print(torch.cuda.get_device_capability())  # (9,0) / (10,0) / (12,0)
```

**注意**：`get_arch_list()` 是「这个轮子能跑哪些架构」，不是「当前机器是哪张卡」。
实际运行某张卡时，PyTorch 会自动挑对应的 SASS（或回退到 PTX JIT）。

---

## 二·补、环境是 CUDA 13.1 怎么办？（常见情况）

**结论：仍然装 `cu130`，不要装 cu132 / cu134。**

### 为什么

PyTorch 官方**只发布三个 CUDA 13 变体**（实测索引探测）：

| 索引 | HTTP | 说明 |
|---|---|---|
| `cu130` | 200 | ✅ 有 torch 2.14.1 |
| `cu131` | **403** | ❌ **不存在** |
| `cu132` | 200 | ✅ 有 torch 2.14.1 |
| `cu133` | 403 | ❌ 不存在 |
| `cu134` | 200 | ✅ 有 torch 2.14.1 |

所以 13.1 没有对应轮子，**只能选邻近版本**。

### 选 cu130 而不是 cu132 的依据：CUDA 次版本兼容（minor version compatibility）

- **规则**：用较旧 CUDA 次版本编译的二进制，可以在**较新**次版本的驱动/运行时上运行（无需重编）。
- cu130 编译 → 跑在 **13.1** 环境 ✅
- cu132 编译 → 需要支持 **13.2** 的驱动；13.1 环境**不满足** ❌
- cu134 同理 ❌

> 一句话：**「用旧的能跑在新的上，用新的跑不了在旧的上面」。**

### 怎么确认机器上到底是哪个版本

这两个不是一回事，务必分清：

```bash
nvidia-smi | grep "CUDA Version"   # 驱动支持的最高 CUDA 版本（能力上限）
nvcc --version                      # 本机 CUDA Toolkit 版本（编译用）
python -c "import torch; print(torch.version.cuda)"   # 轮子自带的运行时版本
```

**判断规则**：`torch.version.cuda`（13.0）必须 ≤ `nvidia-smi` 显示的 CUDA Version（13.1）→ 满足 ✅

### 两个补充提示

1. **如果只是跑 PyTorch（不编译自定义算子）** —— 直接用 cu130，无任何问题。
2. **如果要编译自定义 CUDA 扩展**（`torch.utils.cpp_extension`）：
   - 用 toolkit 13.1 编译、PyTorch 自带 13.0 运行时 —— **通常没问题**（次版本向前兼容，13.1 很少新增 libcudart 符号）
   - 若遇到 `undefined symbol` / 版本不匹配：把 `CUDA_HOME` 指向 13.1 并让 `LD_LIBRARY_PATH` 优先使用系统 CUDA 库，或改用 toolkit 13.0 编译
   - 更保守的做法：扩展也用 13.0 编译，与轮子完全对齐

---

## 三、什么情况下才真的需要自己编译

只有以下情况才需要源码构建：

1. 需要轮子里**没有的架构**（如 sm_103 / sm_110 / sm_121）
2. 需要**裁剪**：只要 3 个架构以减小体积（官方包 554 MB，裁剪后可能 250 MB 左右）
3. 需要**自定义 CUDA 版本**（13.2 / 13.4 等非官方组合）
4. 需要 `USE_*` 特殊开关（如自定义 NCCL、cuDNN、MKL、分布式后端）
5. 目标平台不在官方列表（如某些国产加速卡 —— 那要走厂商 fork）

---

## 四、源码编译——完整资源清单

### 4.1 硬件

| 资源 | 最低 | 推荐 |
|---|---|---|
| CPU 核数 | 16 | 32+（`MAX_JOBS` 决定并行度） |
| 内存 | 32 GB | **64~128 GB**（链接阶段极易 OOM） |
| 磁盘 | 30 GB | **80 GB+**（源码 ~2GB + 构建中间件 ~20GB + wheel ~1GB） |
| GPU | 任意 sm_75+ | 与目标架构一致的卡（仅用于跑测试，编译本身不需要 GPU） |

### 4.2 系统与工具链

| 组件 | 版本要求 | 说明 |
|---|---|---|
| OS | Ubuntu 22.04 / 24.04 | 官方用 manylinux_2_28 容器，glibc ≥ 2.28 |
| CUDA Toolkit | **13.0** | 必须与目标一致 |
| cuDNN | 9.x（CUDA 13 版） | 解压到 `/usr/local/cuda` |
| NCCL | 2.2x（CUDA 13 版） | 多卡必需 |
| GCC | 11 ~ 13 | CUDA 13 支持范围；`gcc-13` 较稳 |
| CMake | ≥ 3.27 | `cmake --version` |
| Ninja | 任意近期版本 | 强烈建议（比 make 快很多） |
| Python | 3.10 ~ 3.13 + **dev headers** | `python3-dev` |
| 其它 | git、curl、unzip、patchelf、pkg-config、libopenblas-dev、libomp-dev | — |

### 4.3 Python 构建依赖

```bash
pip install -r requirements.txt          # PyTorch 源码根目录下
# 关键包：setuptools、wheel、ninja、numpy、pyyaml、typing_extensions、cmake、requests
```

### 4.4 源码

```bash
git clone --recursive --depth 1 --branch v2.14.1 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule sync --recursive
git submodule update --init --recursive --depth 1
```

> `--recursive` 必须：PyTorch 有几十个子模块（kineto、fbgemm、oneDNN、NNPACK…）。

### 4.5 关键环境变量

```bash
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"   # ← 核心：只要这三个架构
export USE_CUDA=1
export USE_NCCL=1
export MAX_JOBS=$(nproc)                       # 内存不足时调小，例如 8
export FORCE_CUDA=1
```

> - `TORCH_CUDA_ARCH_LIST` 可用 `9.0+PTX` 形式附带 PTX。
> - 多个架构用 `;` 分隔；不需要的架构**不要写**，否则体积暴涨、编译变慢。

### 4.6 构建命令

```bash
python setup.py bdist_wheel
# 产物：dist/torch-2.14.1+cu130-cp312-cp312-linux_x86_64.whl
```

**耗时**：32 核 / 64 GB 约 **2~4 小时**；16 核可能 6 小时以上。

### 4.7 最可靠的方式：Docker

```bash
docker run --gpus all -it --rm \
  -v "$PWD:/work" -w /work \
  -v /usr/local/cuda:/usr/local/cuda \
  nvidia/cuda:13.0.0-devel-ubuntu24.04 bash
```

在容器里装依赖后构建，可以避免宿主机的 glibc / 编译器版本问题。
（附带的 `build_from_source.sh` 是可直接跑的骨架脚本。）

---

## 五、常见坑

| 坑 | 现象 | 处理 |
|---|---|---|
| **CUDA 13 驱动版本不足** | `RuntimeError: CUDA driver version is insufficient` | 升级到 ≥580.x |
| **TORCH_CUDA_ARCH_LIST 没设** | 构建脚本报 `must be defined` | 显式 export |
| **内存不足** | 链接阶段被 OOM Killer 杀掉 | 降低 `MAX_JOBS`（如 8）或加 swap |
| **子模块没拉全** | 编译缺头文件（如 kineto / fbgemm） | `git submodule update --init --recursive` |
| **glibc 太低** | 装上去后 `GLIBC_2.28 not found` | 在 manylinux_2_28 或 Ubuntu 22.04+ 构建 |
| **多架构体积爆炸** | wheel 从 550MB 涨到 1GB+ | 只写需要的架构；别加 `+PTX`（除非要前向兼容） |
| **cuDNN/NCCL 版本不匹配** | 运行时报找不到符号 | 用和 CUDA 13.0 匹配的版本 |
| **aarch64 构建** | 官方有 aarch64 轮子，但本地交叉编译很麻烦 | 优先用官方 aarch64 wheel 或 NVIDIA NGC 镜像 |
| **多卡通信异常** | NCCL 超时 | 版本对齐 + 设置 `NCCL_DEBUG=INFO` 排查 |
| **PTX 与驱动** | 新架构上首次运行很慢 | 属正常：PTX 在首次运行 JIT 编译 |

---

## 六、快速判断：你到底该走哪条路

```
你的架构需求 ⊆ {sm_75, sm_80, sm_86, sm_90, sm_100, sm_120} ？
├── 是 → 用官方 cu130 轮子（本文档第二节），5 分钟搞定 ✅
└── 否 → 需要源码编译（第四节），准备 2~6 小时 + 一台大机器
```

以你的需求（**sm90 + sm100 + sm120**）判断：**属于「是」，直接装官方轮子。**

---

## 七、文件说明

| 文件 | 用途 |
|---|---|
| `cuda13_pytorch_notes.md` | 本文件（环境说明） |
| [`scripts/env/install_official.sh`](scripts/env/install_official.sh) | 一条命令安装官方 cu130 轮子（自动查 Python/glibc/驱动，装完自检） |
| [`scripts/env/verify_arch.sh`](scripts/env/verify_arch.sh) | 验证已装环境的架构；也支持**离线检查任意 `.whl`** |
| [`scripts/env/build_from_source.sh`](scripts/env/build_from_source.sh) | 源码编译骨架（仅在确实需要自编译时用） |

---

## 附：参考链接

- PyTorch 官方 wheel 索引（cu130）：<https://download.pytorch.org/whl/cu130/>
- PyTorch 各版本 CUDA 支持矩阵：<https://github.com/pytorch/pytorch/blob/main/RELEASE.md>
- 安装向导：<https://pytorch.org/get-started/locally/>
- CUDA Toolkit 13.0 发行说明：<https://docs.nvidia.com/cuda/archive/13.0.0/>
- PyTorch 源码构建：<https://github.com/pytorch/pytorch#from-source>
