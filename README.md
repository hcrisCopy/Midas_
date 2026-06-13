# MiDaS v2.1 Small Torch GPU 推理交接说明

本交接文件夹只保留 **MiDaS v2.1 small**，也就是作者发布的最小 PyTorch 权重 `midas_v21_small_256.pt`。当前交接只考虑 **GPU + CUDA 12.4**，不提供 CPU、ONNX、OpenVINO、MiDaS v3/v3.1 或其它大模型路径。

目标是：只上传 `MyCode` 到 GitHub；权重不上传，按本文下载到 `MyCode` 同级的 `weights/`；接手者即可在 CUDA 12.4 GPU 环境中跑通 `data/` 图片的 Torch 推理，并能继续基于 small 模型代码改训练。

## 1. 模型版本

本交接包固定使用：

- `model_type`: `midas_v21_small_256`
- 权重: `midas_v21_small_256.pt`
- encoder: `efficientnet_lite3`
- decoder: MiDaS small 的 RefineNet 风格融合结构
- 默认输入上界: `256x256`，保持长宽比并对齐到 32 的倍数
- 运行环境: PyTorch `2.5.1` + CUDA runtime `12.4`

MiDaS 输出的是单目相对深度，不是带真实尺度的米制深度。不同图片之间的数值尺度不能直接当作真实距离比较。

`midas/` 中只保留 small 推理和后续训练改造需要的代码，避免交接时混入其它模型版本。

## 2. 目录结构

建议交接后的目录保持如下：

```text
MiDaS-handoff/
|-- MyCode/
|   |-- README.md
|   |-- requirements.txt
|   |-- infer.py
|   |-- download_weights.py
|   |-- utils.py
|   |-- midas/
|   |   |-- base_model.py
|   |   |-- blocks.py
|   |   |-- midas_net_custom.py
|   |   |-- model_loader.py
|   |   `-- transforms.py
|   |-- data/
|   |   |-- 1.png
|   |   |-- 2.png
|   |   |-- 3.png
|   |   |-- 4.png
|   |   `-- 5.png
|   `-- output/              # 推理后自动生成，不需要上传
`-- weights/                 # 和 MyCode 同级，不上传 GitHub
    `-- midas_v21_small_256.pt
```

默认权重路径是 `../weights/midas_v21_small_256.pt`。`MyCode/.gitignore` 已忽略 `output/`、`*.pt`、`*.onnx` 等生成物和大文件。

## 3. 创建环境

在 Anaconda Prompt 中进入 `MyCode`：

```bat
cd /d D:\your_path\MiDaS-handoff\MyCode
```

创建新的 Conda 环境：

```bat
conda create -n midas-v21-small-cu124 python=3.10.8
conda deactivate
conda activate midas-v21-small-cu124
```

安装支持 CUDA 12.4 的 PyTorch：

```bat
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

安装项目依赖：

```bat
pip install -r requirements.txt
```

检查 CUDA 是否可用：

```bat
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

正常应看到 `torch.version.cuda` 为 `12.4`，`torch.cuda.is_available()` 为 `True`。

`requirements.txt` 不包含 `torch/torchvision`，避免 pip 自动安装 CPU 版 torch。

## 4. 下载权重

在 `MyCode` 目录运行：

```bat
python download_weights.py
```

下载后应得到：

```text
../weights/midas_v21_small_256.pt
```

如果网络无法下载，也可以手动下载：

```text
https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt
```

然后把文件放到 `MyCode` 同级的 `weights/` 文件夹中。

## 5. 跑 GPU 推理

在 `MyCode` 目录运行：

```bat
python infer.py
```

默认行为：

- 输入: `data/` 下的图片
- 模型: 固定 `midas_v21_small_256`
- 设备: 固定 `cuda`
- 权重: `../weights/midas_v21_small_256.pt`
- 输出: `output/`

也可以显式指定路径：

```bat
python infer.py --input_path data --output_path output --model_weights ../weights/midas_v21_small_256.pt
```

输出示例：

```text
output/1-midas_v21_small_256.png
output/1-midas_v21_small_256.pfm
```

`.png` 是归一化后的可视化深度图，默认使用 inferno colormap；`.pfm` 是 float32 原始相对深度输出，更适合后续训练、评估或数值处理。

本机已用 CUDA 12.4 环境验证 `data/1.png` 到 `data/5.png` 可以跑通，生成 5 个 `.png` 和 5 个 `.pfm`。PFM 均为 float32，尺寸与原图一致，数值均为有限值。

常用参数：

```bat
python infer.py --side
python infer.py --grayscale
python infer.py --optimize
python infer.py --height 256
```

`--side` 输出原图和深度图拼接图；`--grayscale` 写 16-bit 灰度 PNG；`--optimize` 在 CUDA 上启用 half-float 推理；`--height` 覆盖 small 模型默认输入高度，一般保持默认即可。

## 6. 推理逻辑

`infer.py` 保持作者 `run.py` 中 MiDaS v2.1 small 的 Torch 推理链路：

1. `utils.read_image` 用 OpenCV 读图，BGR 转 RGB，并缩放到 `[0, 1]`。
2. `midas.model_loader.load_model` 创建 `MidasNet_small` 和预处理 transform。
3. transform 顺序是 `Resize -> NormalizeImage -> PrepareForNet`。
4. `torch.from_numpy(image).to(device).unsqueeze(0)` 得到 batch 输入。
5. `model.forward(sample)` 前向推理。
6. `torch.nn.functional.interpolate(..., mode="bicubic", align_corners=False)` 插值回原图大小。
7. `utils.write_depth` 写可视化 PNG，`utils.write_pfm` 写 float32 PFM。

没有改 small 模型的层结构、forward 逻辑或权重含义。唯一兼容性小补丁在 `midas/blocks.py`：给 `torch.hub.load(...)` 增加 `trust_repo=True, skip_validation=True`，用于避免 PyTorch Hub 访问 GitHub API 时触发 403 rate limit。加载的仍然是原逻辑指定的 EfficientNet-Lite3 hub 模型。

## 7. 后续改训练代码

当前交接包只提供推理脚本，但保留了 small 模型训练改造需要的核心 PyTorch 代码：

- 模型构造入口: `midas/model_loader.py`
- small 模型主体: `midas/midas_net_custom.py`
- encoder/decoder block: `midas/blocks.py`
- 图像 transform: `midas/transforms.py`
- 权重读取: `midas/base_model.py`

改训练时通常从 `infer.py` 拆出这些部分：

1. 用自己的 Dataset 读取 `image` 和监督信号 `depth/disparity`。
2. 复用 `load_model(...)` 创建 `MidasNet_small`，或直接实例化 `MidasNet_small`。
3. 去掉 `torch.no_grad()`，把 `model.eval()` 改为 `model.train()`。
4. 前向仍然用 `prediction = model.forward(sample)`。
5. 按数据集定义 loss，例如 scale-invariant loss、L1、ranking loss 或自定义相对深度 loss。
6. 用 optimizer 保存 checkpoint。作者的 `BaseModel.load` 能读取普通 state_dict 或包含 `optimizer` 字段的 checkpoint。

## 8. 复查清单

- 是否只考虑 MiDaS v2.1 small: 是，只服务 `midas_v21_small_256`。
- 是否只上传 `MyCode` 就够: 是。代码、README、依赖文件、测试图都在 `MyCode`；权重按 README 下载到同级 `weights/`。
- 是否避免 CPU torch 自动安装: 是。`requirements.txt` 不包含 torch，README 要求先安装 CUDA 12.4 版 PyTorch。
- 是否只使用 GPU/CUDA 12.4: 是。`infer.py` 固定使用 `cuda`，无 CUDA 会直接报错。
- 是否使用 Torch: 是。默认推理使用 `.pt` 权重和 PyTorch 前向，不走 ONNX/OpenVINO。
- 是否下载最小权重: 是，默认下载并使用 `midas_v21_small_256.pt`。
- 是否改了 small 模型结构: 没有。只删除交接暂不需要的其它模型分支，并保留 small 推理和训练改造所需代码。
- 推理逻辑是否和作者 small 分支一致: 是。读图、transform、forward、bicubic 插值、PNG/PFM 输出与作者 `run.py` 的 Torch 路径一致。
