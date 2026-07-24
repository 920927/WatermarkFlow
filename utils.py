"""
水印实验通用工具：传统攻击、评测指标、数据加载、VAE 压缩、报告输出。
供 text_to_image / image_to_image / image_edit 及消融脚本共用。
"""
import glob
import io
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import auc, roc_curve
from torchvision import transforms


# ========================= 基础攻击 =========================

def attack_random_mask(image, param=0.3):
    w, h = image.size
    mask_area = int(w * h * param)
    aspect_ratio = random.uniform(0.5, 2.0)
    mask_height = int((mask_area / aspect_ratio) ** 0.5)
    mask_width = int(mask_height * aspect_ratio)
    mask_width = min(mask_width, w)
    mask_height = min(mask_height, h)
    if mask_width < 10 or mask_height < 10:
        mask_width = min(w, 10)
        mask_height = min(h, 10)

    left = random.randint(0, w - mask_width)
    top = random.randint(0, h - mask_height)
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle(
        [left, top, left + mask_width, top + mask_height], fill=(0, 0, 0)
    )
    return masked


def calculate_psnr(img1, img2, img_size=1024):
    a = np.array(img1.convert("RGB").resize((img_size, img_size))).astype(np.float64)
    b = np.array(img2.convert("RGB").resize((img_size, img_size))).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0


def apply_attack(image, attack_type, param):
    if attack_type == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=param)
        return Image.open(buf)
    if attack_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=param))
    if attack_type == "resize":
        w, h = image.size
        return image.resize((int(w * param), int(h * param))).resize(
            (w, h), Image.Resampling.LANCZOS
        )
    if attack_type == "crop":
        return attack_random_mask(image, param)
    if attack_type == "rotate":
        return image.rotate(param)
    if attack_type == "brightness":
        return ImageEnhance.Brightness(image).enhance(param)
    if attack_type == "gaussian_noise":
        arr = np.array(image)
        return Image.fromarray(np.clip(arr + np.random.normal(0, param, arr.shape), 0, 255).astype(np.uint8))
    if attack_type == "uniform_noise":
        arr = np.array(image)
        return Image.fromarray(np.clip(arr + np.random.uniform(-param, param, arr.shape), 0, 255).astype(np.uint8))
    if attack_type == "salt_pepper_noise":
        arr = np.array(image)
        n = int(param * arr.size)
        for _ in range(n):
            x = random.randint(0, arr.shape[0] - 1)
            y = random.randint(0, arr.shape[1] - 1)
            arr[x, y] = 255 if random.random() < 0.5 else 0
        return Image.fromarray(arr)
    if attack_type == "exponential_noise":
        arr = np.array(image)
        return Image.fromarray(np.clip(arr + np.random.exponential(param, arr.shape), 0, 255).astype(np.uint8))
    if attack_type == "poisson_noise":
        arr = np.array(image)
        noisy = np.random.poisson(arr / 255.0 * param) * 255
        return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))
    if attack_type == "filter":
        return image.filter(ImageFilter.CONTOUR)
    return image


# ========================= 攻击参数表 =========================

TRADITIONAL_ATTACKS = [
    ("jpeg_10", "jpeg", 10), ("jpeg_20", "jpeg", 20), ("jpeg_30", "jpeg", 30),
    ("jpeg_50", "jpeg", 50), ("jpeg_70", "jpeg", 70), ("jpeg_90", "jpeg", 90),
    ("blur_0.5", "blur", 0.5), ("blur_1", "blur", 1), ("blur_2", "blur", 2),
    ("blur_3", "blur", 3), ("blur_5", "blur", 5), ("blur_7", "blur", 7),
    ("resize_0.9", "resize", 0.9), ("resize_0.7", "resize", 0.7),
    ("resize_0.5", "resize", 0.5), ("resize_0.3", "resize", 0.3),
    ("crop_0.05", "crop", 0.05), ("crop_0.15", "crop", 0.15),
    ("crop_0.3", "crop", 0.3), ("crop_0.5", "crop", 0.5),
    ("crop_0.7", "crop", 0.7), ("crop_0.9", "crop", 0.9),
    ("rotate_5", "rotate", 5), ("rotate_25", "rotate", 25),
    ("rotate_45", "rotate", 45), ("rotate_90", "rotate", 90),
    ("brightness_0.2", "brightness", 0.2), ("brightness_0.5", "brightness", 0.5),
    ("brightness_0.8", "brightness", 0.8), ("brightness_1.2", "brightness", 1.2),
    ("brightness_1.5", "brightness", 1.5), ("brightness_1.8", "brightness", 1.8),
    ("brightness_2", "brightness", 2),
    ("gaussian_noise_2", "gaussian_noise", 2), ("gaussian_noise_5", "gaussian_noise", 5),
    ("gaussian_noise_10", "gaussian_noise", 10), ("gaussian_noise_20", "gaussian_noise", 20),
    ("gaussian_noise_30", "gaussian_noise", 30), ("gaussian_noise_50", "gaussian_noise", 50),
    ("uniform_noise_2", "uniform_noise", 2), ("uniform_noise_5", "uniform_noise", 5),
    ("uniform_noise_10", "uniform_noise", 10), ("uniform_noise_20", "uniform_noise", 20),
    ("uniform_noise_30", "uniform_noise", 30), ("uniform_noise_50", "uniform_noise", 50),
    ("salt_pepper_0001", "salt_pepper_noise", 0.001),
    ("salt_pepper_0005", "salt_pepper_noise", 0.005),
    ("salt_pepper_001", "salt_pepper_noise", 0.01),
    ("salt_pepper_002", "salt_pepper_noise", 0.02),
    ("salt_pepper_003", "salt_pepper_noise", 0.03),
    ("salt_pepper_005", "salt_pepper_noise", 0.05),
    ("exp_noise_02", "exponential_noise", 0.2), ("exp_noise_05", "exponential_noise", 0.5),
    ("exp_noise_08", "exponential_noise", 0.8), ("exp_noise_1", "exponential_noise", 1.0),
    ("exp_noise_3", "exponential_noise", 3.0), ("exp_noise_5", "exponential_noise", 5.0),
    ("poisson_2", "poisson_noise", 2), ("poisson_5", "poisson_noise", 5),
    ("poisson_10", "poisson_noise", 10), ("poisson_20", "poisson_noise", 20),
    ("filter", "filter", 0),
]

BIT_ACC_GROUPS = [
    ("JPEG Compression", ["jpeg_10", "jpeg_20", "jpeg_30", "jpeg_50", "jpeg_70", "jpeg_90"]),
    ("Gaussian Blur", ["blur_0.5", "blur_1", "blur_2", "blur_3", "blur_5", "blur_7"]),
    ("Resize", ["resize_0.9", "resize_0.7", "resize_0.5", "resize_0.3"]),
    ("Crop / Mask", ["crop_0.05", "crop_0.15", "crop_0.3", "crop_0.5", "crop_0.7", "crop_0.9"]),
    ("Rotation", ["rotate_5", "rotate_25", "rotate_45", "rotate_90"]),
    ("Brightness", ["brightness_0.2", "brightness_0.5", "brightness_0.8",
                     "brightness_1.2", "brightness_1.5", "brightness_1.8", "brightness_2"]),
    ("Gaussian Noise", ["gaussian_noise_2", "gaussian_noise_5", "gaussian_noise_10",
                        "gaussian_noise_20", "gaussian_noise_30", "gaussian_noise_50"]),
    ("Uniform Noise", ["uniform_noise_2", "uniform_noise_5", "uniform_noise_10",
                       "uniform_noise_20", "uniform_noise_30", "uniform_noise_50"]),
    ("Salt & Pepper", ["salt_pepper_0001", "salt_pepper_0005", "salt_pepper_001",
                       "salt_pepper_002", "salt_pepper_003", "salt_pepper_005"]),
    ("Exponential Noise", ["exp_noise_02", "exp_noise_05", "exp_noise_08",
                           "exp_noise_1", "exp_noise_3", "exp_noise_5"]),
    ("Poisson Noise", ["poisson_2", "poisson_5", "poisson_10", "poisson_20"]),
    ("Filter", ["filter"]),
]


# ========================= 指标 / 数据 =========================

def bit_accuracy(target_bits, bits):
    return sum(a == b for a, b in zip(target_bits, bits)) / len(target_bits)


def list_image_files(input_dir, limit=None):
    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    return files[:limit] if limit is not None else files


def load_coco_prompts(
    image_dir="../dataset/val2017",
    annotations_path="../dataset/annotations/captions_val2017.json",
):
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    id_to_caption = {}
    for anno in annotations_data["annotations"]:
        if anno["image_id"] not in id_to_caption:
            id_to_caption[anno["image_id"]] = anno["caption"]

    labels, prompts = [], []
    for path in sorted(glob.glob(os.path.join(image_dir, "*"))):
        label = int(os.path.splitext(os.path.basename(path))[0])
        if label in id_to_caption:
            labels.append(label)
            prompts.append(id_to_caption[label])
    return labels, prompts


def load_clean_image(label, clean_dir="./output_generate_orig"):
    matches = glob.glob(os.path.join(clean_dir, f"*{int(label)}*"))
    if not matches:
        return None
    return Image.open(matches[0]).convert("RGB")


def load_wbench_edits(
    prompts_csv="../../llm_model/W-Bench/DET_INVERSION_1K/prompts.csv",
    image_dir="../../llm_model/W-Bench/DET_INVERSION_1K/image",
    limit=None,
):
    import pandas as pd

    df = pd.read_csv(prompts_csv)
    if limit is not None:
        df = df.iloc[:limit]
    samples = []
    for _, row in df.iterrows():
        stem = f"{row['idx']}_{row['ID']}"
        samples.append((os.path.join(image_dir, f"{stem}.png"), row["edit_prompt"], stem))
    return samples


def compute_roc_metrics(pos_scores, neg_scores, name=""):
    if not pos_scores or not neg_scores:
        print(f"\n===== {name} =====\n(跳过: 正/负样本不足)")
        return float("nan")

    labels = np.array([1] * len(pos_scores) + [0] * len(neg_scores))
    scores = np.array(pos_scores + neg_scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    tpr_at_1 = np.interp(0.01, fpr, tpr)
    print(f"\n===== {name} =====")
    print(f"AUC: {roc_auc:.4f}")
    print(f"TPR @ 1% FPR: {tpr_at_1:.4f}")
    return roc_auc


def run_traditional_attack(image, name, attack_type, param):
    return name, apply_attack(image, attack_type, param)


def build_vae_networks(device, qualities=(1, 2, 3, 4, 5, 6)):
    from compressai.zoo import (
        bmshj2018_factorized, bmshj2018_hyperprior, mbt2018_mean, mbt2018, cheng2020_anchor,
    )
    builders = {
        "bmshj2018-factorized": bmshj2018_factorized,
        "bmshj2018-hyperprior": bmshj2018_hyperprior,
        "mbt2018-mean": mbt2018_mean,
        "mbt2018": mbt2018,
        "cheng2020-anchor": cheng2020_anchor,
    }
    networks = {}
    for q in qualities:
        for name, builder in builders.items():
            networks[f"{name}_{q}"] = builder(quality=q, pretrained=True).eval().to(device)
    return networks


@torch.no_grad()
def vae_compress(image, net, device, output_size=None):
    x = transforms.ToTensor()(image).unsqueeze(0).to(device)
    out = net(x)
    out["x_hat"].clamp_(0, 1)
    attacked = transforms.ToPILImage()(out["x_hat"].squeeze().cpu())
    if output_size is not None:
        attacked = attacked.resize(output_size, Image.Resampling.LANCZOS)
    return attacked


# ========================= 报告 =========================

def report_psnr(psnr_list):
    if psnr_list:
        print("\n================ Imperceptibility ================")
        print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")


def report_bit_accuracy(bit_stats, n_samples):
    print("\n" + "=" * 60)
    print(f"Bit Accuracy (Samples: {n_samples})")
    print("=" * 60)
    if "none" in bit_stats and bit_stats["none"]:
        print(f"\nNo Attack: {np.mean(bit_stats['none']) * 100:.2f}%")

    for title, keys in BIT_ACC_GROUPS:
        present = [k for k in keys if k in bit_stats and bit_stats[k]]
        if not present:
            continue
        print(f"\n[{title}]")
        for k in present:
            print(f"  {k:<32}: {np.mean(bit_stats[k]) * 100:.2f}%")
        print(f"  {'Average':<32}: {np.mean([np.mean(bit_stats[k]) for k in present]) * 100:.2f}%")

    grouped = {k for _, keys in BIT_ACC_GROUPS for k in keys} | {"none"}
    extras = [k for k in bit_stats if k not in grouped and bit_stats[k]]
    if extras:
        print("\n[Other / VAE]")
        for k in extras:
            print(f"  {k:<32}: {np.mean(bit_stats[k]) * 100:.2f}%")
    print("\n" + "=" * 60)


def report_detection(stats):
    print("\n================ Detection =================")
    compute_roc_metrics(stats["score_wm"], stats["score_clean"], "Original")
    if stats["score_attacks"]:
        print("\n================ Robustness AUC ================")
        aucs = [
            compute_roc_metrics(scores, stats["score_clean"], atk)
            for atk, scores in stats["score_attacks"].items()
        ]
        valid = [a for a in aucs if not np.isnan(a)]
        if valid:
            print("\nAverage Robustness AUC:", np.mean(valid))
    if stats["det_acc"]:
        print("\n================ Detection Acc by Attack ================")
        for atk, vals in stats["det_acc"].items():
            print(f"{atk:<30}: {np.mean(vals) * 100:.2f}%")
