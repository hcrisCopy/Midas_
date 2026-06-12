# MiDaS Torch 推理交接说明

这个文件夹是从原 MiDaS 项目中整理出的 Torch 推理交接包。目标是：只把 `MyCode` 上传到 GitHub，交接者按本 README 配好环境、下载权重，就能用 `data/` 里的图片跑出深度图；后续也可以基于这里保留的 PyTorch 模型源码改训练代码。

## 项目模型理解

MiDaS 做的是单目相对深度估计：输入一张 RGB 图，输出同尺寸的相对深度/视差响应。它不是相机标定后的绝对米制深度，所以不同图片之间的数值尺度不能直接当作真实距离比较。

这个代码库里主要有两类 PyTorch 模型：

- MiDaS v2.1 CNN 系列：`midas_v21_384` 使用 ResNeXt101-WSL encoder；`midas_v21_small_256` 使用 EfficientNet-Lite3 encoder，是默认交接使用的最小权重。对应代码在 `midas/midas_net.py` 和 `midas/midas_net_custom.py`。
- MiDaS v3/v3.1 DPT Transformer 系列：BEiT、Swin/SwinV2、LeViT、ViT、Next-ViT 等 backbone，加上 DPT 风格 dense prediction decoder。对应代码在 `midas/dpt_depth.py`、`midas/blocks.py`、`midas/backbones/`。

统一入口是 `midas/model_loader.py`。它根据 `model_type` 决定 backbone、输入分辨率、是否保持长宽比、归一化参数和 transform。推理入口 `infer.py` 只调用这个统一入口，不重新实现模型。

## 1. 目录结构

建议交接后的目录保持下面这样：

```text
MiDaS-handoff/
├─ MyCode/
│  ├─ README.md
│  ├─ infer.py
│  ├─ download_weights.py
│  ├─ environment-cpu.yml
│  ├─ environment-cu117.yml
│  ├─ utils.py
│  ├─ midas/
│  │  ├─ model_loader.py
│  │  ├─ dpt_depth.py
│  │  ├─ midas_net.py
│  │  ├─ midas_net_custom.py
│  │  ├─ blocks.py
│  │  └─ backbones/
│  ├─ data/
│  │  ├─ 1.png
│  │  ├─ 2.png
│  │  ├─ 3.png
│  │  ├─ 4.png
│  │  └─ 5.png
│  └─ output/              # 推理后自动生成，不需要上传
└─ weights/                # 和 MyCode 同级，不上传 GitHub
   └─ midas_v21_small_256.pt
```

`weights/` 不能放进 GitHub；`.gitignore` 已经忽略 `*.pt`、`output/` 等生成物。默认权重路径是 `../weights/midas_v21_small_256.pt`，即 `MyCode` 的同级目录。

## 2. 创建环境

在 Anaconda Prompt 里进入 `MyCode`：

```bat
cd /d D:\your_path\MiDaS-handoff\MyCode
```

没有 NVIDIA GPU 或只想先跑通 CPU：

```bat
conda env create -f environment-cpu.yml
conda activate midas-mycode
```

有 NVIDIA GPU 且本机驱动支持 CUDA 11.7：

```bat
conda env create -f environment-cu117.yml
conda activate midas-mycode-cu117
```

依赖版本沿用作者项目的核心组合：Python 3.10.8、PyTorch 1.13.0、torchvision 0.14.0、timm 0.6.12、opencv-python 4.6.0.66、einops 0.6.0。第一次运行 `midas_v21_small_256` 时，作者代码会通过 `torch.hub` 加载 EfficientNet-Lite3 backbone 代码，这是原始 PyTorch 实现的一部分。

## 3. 下载最小权重

默认使用作者提供的最小 Torch 权重 `midas_v21_small_256.pt`，参数量约 21M，适合先跑通交接。

在 `MyCode` 目录运行：

```bat
python download_weights.py --model_type midas_v21_small_256
```

下载后应得到：

```text
../weights/midas_v21_small_256.pt
```

如果网络无法下载，也可以手动浏览器下载：

```text
https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
```

然后把文件放到 `MyCode` 同级的 `weights/` 文件夹中。

## 4. 跑通推理

在 `MyCode` 目录运行：

```bat
python infer.py
```

默认行为：

- 输入：`data/` 下所有常见图片格式。
- 模型：`midas_v21_small_256`。
- 权重：`../weights/midas_v21_small_256.pt`。
- 输出：`output/`。

也可以显式写全：

```bat
python infer.py --input_path data --output_path output --model_type midas_v21_small_256 --model_weights ../weights/midas_v21_small_256.pt
```

输出文件示例：

```text
output/1-midas_v21_small_256.png
output/1-midas_v21_small_256.pfm
```

`.png` 是归一化后的可视化深度图，默认使用 inferno colormap；`.pfm` 是 float32 原始相对深度输出，更适合后续训练、评估或数值处理。MiDaS 输出的是相对深度，不是带真实尺度的米制深度。

本机已用 `environment-cpu.yml` 创建的 `midas-mycode` 环境验证：

```text
python infer.py --device cpu
Model loaded, number of parameters = 21M
处理 data/1.png 到 data/5.png 共 5 张图片
生成 output/ 下 5 个 .png 和 5 个 .pfm
PFM 均为 float32，尺寸与原图一致，数值均为有限值
```

常用参数：

```bat
python infer.py --side
python infer.py --grayscale
python infer.py --device cpu
python infer.py --device cuda --optimize
python infer.py --height 384
python infer.py --square
```

`--side` 会输出原图和深度图拼接图；`--grayscale` 会写 16-bit 灰度 PNG；`--height` 和 `--square` 与作者原始 `run.py` 参数含义一致。

如果直接用系统默认 Python 运行时遇到 `timm.models.maxxvit` 的 dataclass 报错，通常是 Python 版本过新导致 `timm==0.6.12` 不兼容。按本 README 使用 Python 3.10 的 conda 环境即可。

## 5. 推理逻辑是否和原作者一致

`infer.py` 保持作者 `run.py` 的 Torch 推理链路：

1. `utils.read_image` 用 OpenCV 读图，BGR 转 RGB，并缩放到 `[0, 1]`。
2. `midas.model_loader.load_model` 根据 `model_type` 创建模型和预处理 transform。
3. transform 顺序是作者原来的 `Resize -> NormalizeImage -> PrepareForNet`。
4. `torch.from_numpy(image).to(device).unsqueeze(0)` 得到 batch 输入。
5. `model.forward(sample)` 前向推理。
6. `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)` 插值回原图大小。
7. `utils.write_depth` 写可视化 PNG，`utils.write_pfm` 写 float32 PFM。

没有改动 `midas/` 内部模型结构、backbone、decoder、预处理或权重含义。这里新增的主要是路径更清晰的交接入口和下载脚本。

唯一的兼容性小补丁在 `midas/blocks.py`：给 `torch.hub.load(...)` 增加了 `trust_repo=True, skip_validation=True`，用于避免 PyTorch Hub 访问 GitHub API 时触发 403 rate limit。下载和加载的仍然是作者原逻辑指定的同一个 EfficientNet-Lite3 或 ResNeXt hub 模型，不改变网络结构或推理结果。

## 6. 作者模型版本说明

`MyCode/midas/model_loader.py` 保留了作者给出的全部 Torch 模型版本代码，默认权重都放在 `../weights/`。下表是常用 `model_type` 与下载链接。

| model_type | 版本/结构 | 默认输入 | 权重链接 |
|---|---:|---:|---|
| `dpt_beit_large_512` | MiDaS 3.1 BEiT-L | 512 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt |
| `dpt_beit_large_384` | MiDaS 3.1 BEiT-L | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt |
| `dpt_beit_base_384` | MiDaS 3.1 BEiT-B | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt |
| `dpt_swin2_large_384` | MiDaS 3.1 SwinV2-L | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt |
| `dpt_swin2_base_384` | MiDaS 3.1 SwinV2-B | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt |
| `dpt_swin2_tiny_256` | MiDaS 3.1 SwinV2-T | 256 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt |
| `dpt_swin_large_384` | MiDaS 3.1 Swin-L | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt |
| `dpt_next_vit_large_384` | MiDaS 3.1 Next-ViT-L | 384 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt |
| `dpt_levit_224` | MiDaS 3.1 LeViT | 224 | https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt |
| `dpt_large_384` | MiDaS 3.0 DPT-L | 384 | https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt |
| `dpt_hybrid_384` | MiDaS 3.0 DPT-Hybrid | 384 | https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt |
| `midas_v21_384` | MiDaS 2.1 Large | 384 | https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt |
| `midas_v21_small_256` | MiDaS 2.1 Small | 256 | https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt |

切换模型示例：

```bat
python download_weights.py --model_type dpt_swin2_tiny_256
python infer.py --model_type dpt_swin2_tiny_256
```

注意：

- `dpt_next_vit_large_384` 还需要作者 README 中提到的 Next-ViT 外部代码。可在 `MyCode` 目录执行：`git clone https://github.com/isl-org/Next-ViT midas/external/next_vit`。
- 作者还提供 `openvino_midas_v21_small_256`，它不是 Torch `.pt` 推理，不属于本交接默认路径。如确需 OpenVINO，需要额外安装 `openvino` 并同时下载 `.xml` 和 `.bin`。

## 7. 后续改训练代码的入口

这份交接包不是训练脚本，但保留了训练改造需要的核心 PyTorch 模型代码：

- 模型构造入口：`midas/model_loader.py`。
- DPT 模型：`midas/dpt_depth.py`。
- MiDaS v2.1 大模型：`midas/midas_net.py`。
- MiDaS v2.1 small 模型：`midas/midas_net_custom.py`。
- encoder/decoder block：`midas/blocks.py` 和 `midas/backbones/`。
- 图像 transform：`midas/transforms.py`。

改训练时通常从 `infer.py` 中拆出下面几块：

1. 用自己的 Dataset 读取 `image` 和监督信号 `depth/disparity`。
2. 复用 `load_model(...)` 创建模型，或直接实例化 `MidasNet_small`/`DPTDepthModel`。
3. 去掉 `torch.no_grad()` 和 `model.eval()`，改为 `model.train()`。
4. 前向仍然用 `prediction = model.forward(sample)`。
5. 按数据集定义 loss，例如 scale-invariant loss、L1、ranking loss 或自定义相对深度 loss。
6. 用 optimizer 保存 checkpoint。作者 `BaseModel.load` 能读取普通 state_dict 或包含 `optimizer` 字段的 checkpoint。

## 8. 交接复查清单

- 是否只上传 `MyCode` 就够：是。代码、README、环境文件、测试图片都在 `MyCode`；权重按 README 下载到同级 `weights/`。
- 是否使用 Torch：是。默认推理使用 `.pt` 权重和 PyTorch 前向，不走 ONNX。
- 是否下载了最小权重：默认下载并使用 `midas_v21_small_256.pt`。
- 是否改了原模型结构：没有。`midas/` 保留作者 Torch 模型源码；仅对 `torch.hub.load` 增加跳过 GitHub API 校验的兼容参数，不改变层、forward 或权重。
- 是否保留全部作者模型版本代码：是。`model_loader.py` 中所有作者模型分支都保留，README 列出了全部 Torch 模型下载链接。
- 推理逻辑是否严格一致：是。读图、transform、forward、bicubic 插值、PNG/PFM 输出与作者 `run.py` 的 Torch 路径一致。
