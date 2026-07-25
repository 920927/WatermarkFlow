# WatermarkFlow

Flow-matching diffusion 水印方法复现代码（SD3 / FLUX）。

在去噪轨迹上注入径向对称频域基，支持嵌入、提取与检测，并提供文生图 / 图生图 / 图像编辑评测脚本。

## Directory Structure

```
WatermarkFlow/
├── watermarker.py              # 核心水印器（SD3 / FLUX）
├── utils.py                    # 攻击、指标、数据加载
├── evaluate_robustness.py      # 简易鲁棒性评测
├── t2i_quality.py              # 文生图质量（CLIP / FID / IS）
├── i2i_quality.py              # 图生图质量（PSNR / SSIM / LPIPS）
├── SD3_watermark/              # Stable Diffusion 3 实验入口
│   ├── text_to_image.py
│   ├── image_to_image.py
│   └── image_edit.py
├── FLUX_watermark/             # FLUX 实验入口
│   ├── text_to_image.py
│   ├── image_to_image.py
│   ├── image_edit.py
│   ├── generate_t2i_batch.py
│   └── run_t2i_1000.sh
├── requirements.txt
└── README.md
```

## Installation

```bash
cd WatermarkFlow
conda create -n watermarkflow python=3.10 -y
conda activate watermarkflow

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

## 数据与模型路径

默认路径均为相对路径，请在 `WatermarkFlow/` 目录下运行脚本。若布局不同，可用命令行参数覆盖。

| 资源 | 默认相对路径 |
|------|----------------|
| COCO val2017 图像 | `../dataset/val2017` |
| COCO captions | `../dataset/annotations/captions_val2017.json` |
| SD3 权重 | `../../llm_model/stable-diffusion-3-medium-diffusers` |
| FLUX.1-dev 权重 | `../../llm_model/FLUX.1-dev` |
| W-Bench 编辑数据 | `../../llm_model/W-Bench/DET_INVERSION_1K/` |

也可自行将模型软链到上述相对位置：

```bash
mkdir -p ../../llm_model
# 将本地 SD3 / FLUX 目录软链到 ../../llm_model/ 下对应名称
```

## 快速开始

以下命令均在 `WatermarkFlow/` 下执行。

### SD3 图生图水印

```bash
python SD3_watermark/image_to_image.py \
  --input_dir ../dataset/val2017 \
  --output_dir ./output_track \
  --model_path ../../llm_model/stable-diffusion-3-medium-diffusers \
  --img_num 10 \
  --message SDFLOW \
  --metric both \
  --attacks traditional
```

### SD3 文生图水印

```bash
python SD3_watermark/text_to_image.py \
  --output_dir ./output_generate \
  --model_path ../../llm_model/stable-diffusion-3-medium-diffusers \
  --img_num 10 \
  --message SDFLOW \
  --metric both \
  --attacks traditional
```

### SD3 图像编辑水印

```bash
# 单张 demo
python SD3_watermark/image_edit.py --demo \
  --model_path ../../llm_model/stable-diffusion-3-medium-diffusers \
  --img_in ./000000001584.jpg

# 批量（需 W-Bench）
python SD3_watermark/image_edit.py \
  --output_dir ./output_edit \
  --img_num 10 \
  --metric both \
  --attacks traditional
```

### FLUX

```bash
python FLUX_watermark/image_to_image.py --img_num 10 --metric both --attacks traditional
python FLUX_watermark/text_to_image.py --img_num 10 --metric both --attacks traditional
python FLUX_watermark/image_edit.py --demo --img_in ./000000001584.jpg
```

大批量文生图流水线（生成 → 鲁棒性 → 质量）：

```bash
bash FLUX_watermark/run_t2i_1000.sh
```

### 质量与鲁棒性

```bash
# 图生图质量
python i2i_quality.py \
  --orig_dir ../dataset/val2017 \
  --watermarked_dir ./output_track

# 文生图质量
python t2i_quality.py \
  --wm_dir ./output_generate \
  --real_dir ./output_generate_orig \
  --metrics all

# 简易鲁棒性
python evaluate_robustness.py
```

## 主要参数

| 参数 | 说明 |
|------|------|
| `--strength` | 轨迹扰动强度 |
| `--denoising_strength` | img2img / edit 去噪强度 |
| `--message` | 水印字符串（默认 `SDFLOW`） |
| `--metric` | `bit` / `detect` / `both` |
| `--attacks` | `traditional` / `vae` / `all`（VAE 需 `compressai`） |
| `--img_num` | 评测样本数 |
| `--gpu_ids` | `CUDA_VISIBLE_DEVICES` |

## 核心类（`watermarker.py`）

- `SD3FlowTrajectoryWatermarker`：SD3 图生图轨迹水印
- `SD3Text2ImgWatermarker`：SD3 文生图水印
- `SD3ImgEditWatermarker`：SD3 图像编辑水印
- `FluxText2ImgWatermarker` / `FluxImg2ImgWatermarker` / `FluxImgEditWatermarker`：FLUX 对应实现

## 说明

- 请从本目录运行脚本，以保证相对路径默认值正确。
- 未包含实验结果目录（`output_*`）、草稿（`draft/`）、消融脚本等与主方法复现无关的内容。
- VAE 压缩攻击首次运行时会通过 `compressai` 自动下载预训练权重。
