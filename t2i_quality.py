"""
文生图场景水印图像质量度量（CLIP Score / FID / Inception Score）

用法示例:
  python t2i_quality.py --wm_dir ./FLUX_watermark/output_flux_t2i --metrics all --img_num 4
"""
import argparse
import glob
import json
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torchvision.models import inception_v3
from tqdm import tqdm


# ===================== CLIP Score =====================

def load_clip(device):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def compute_clip_score(model, preprocess, text, image_path, device):
    text_token = clip.tokenize([text], truncate=True).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text_token)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
    return (image_feature @ text_feature.T).item()


def load_prompts_for_wm_dir(wm_dir, annotations_path, image_dir):
    """按水印图文件名中的 COCO image_id 匹配 caption。"""
    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations_data = json.load(f)

    id_to_caption = {}
    for anno in annotations_data["annotations"]:
        if anno["image_id"] not in id_to_caption:
            id_to_caption[anno["image_id"]] = anno["caption"]

    pairs = []
    for path in sorted(glob.glob(os.path.join(wm_dir, "*"))):
        if not path.lower().endswith(("png", "jpg", "jpeg")):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        # 支持 000000000139_w / 139_w / 000000000139
        label_str = stem.replace("_w", "").replace("_orig", "")
        try:
            label = int(label_str)
        except ValueError:
            continue
        if label in id_to_caption:
            pairs.append((path, id_to_caption[label], label))
    return pairs


def evaluate_clip_score(wm_dir, annotations_path, image_dir, img_num=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(device)
    pairs = load_prompts_for_wm_dir(wm_dir, annotations_path, image_dir)
    if img_num is not None:
        pairs = pairs[:img_num]
    if not pairs:
        raise RuntimeError(f"No matched image-prompt pairs in {wm_dir}")

    scores = []
    for path, prompt, _ in tqdm(pairs, desc="CLIP Score"):
        scores.append(compute_clip_score(model, preprocess, prompt, path, device))
    return float(np.mean(scores)), scores


# ===================== FID =====================

class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        model = inception_v3(weights="DEFAULT", transform_input=False)
        model.fc = torch.nn.Identity()
        self.model = model.to(device).eval()

    def forward(self, x):
        return self.model(x)


FID_TRANSFORM = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize([0.5] * 3, [0.5] * 3),
])


def load_images_as_tensors(folder, transform=FID_TRANSFORM):
    images = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(("png", "jpg", "jpeg")):
            continue
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        images.append(transform(img))
    if not images:
        raise RuntimeError(f"No images found in {folder}")
    return torch.stack(images)


@torch.no_grad()
def get_features(images, model, device, batch_size=32):
    feats = []
    for i in tqdm(range(0, len(images), batch_size), desc="FID features"):
        batch = images[i:i + batch_size].to(device)
        feats.append(model(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def calculate_fid(real_dir, fake_dir, device=None, batch_size=32):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionFeatureExtractor(device)
    real_imgs = load_images_as_tensors(real_dir)
    fake_imgs = load_images_as_tensors(fake_dir)
    real_feats = get_features(real_imgs, model, device, batch_size)
    fake_feats = get_features(fake_imgs, model, device, batch_size)

    mu1, sigma1 = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    # 样本过少时 cov 可能是标量
    if np.ndim(sigma1) == 0:
        sigma1 = np.array([[sigma1]])
        sigma2 = np.array([[sigma2]])
        real_feats = real_feats.reshape(-1, 1)
        fake_feats = fake_feats.reshape(-1, 1)
        mu1, mu2 = real_feats.mean(axis=0), fake_feats.mean(axis=0)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


# ===================== Inception Score =====================

def load_pil_images(folder):
    images = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(("png", "jpg", "jpeg")):
            continue
        try:
            images.append(Image.open(os.path.join(folder, fname)).convert("RGB"))
        except Exception:
            pass
    return images


def get_inception_preds(model, images, device, batch_size=32):
    preprocess = T.Compose([
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    preds = []
    for i in tqdm(range(0, len(images), batch_size), desc="Inception preds"):
        batch = torch.stack([preprocess(img) for img in images[i:i + batch_size]]).to(device)
        with torch.no_grad():
            probs = F.softmax(model(batch), dim=1)
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds, axis=0)


def calculate_inception_score(image_folder, splits=10, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = load_pil_images(image_folder)
    if not images:
        raise RuntimeError(f"No images found in {image_folder}")

    # 小样本时自动降低 splits，避免空 split
    splits = max(1, min(splits, len(images)))

    model = inception_v3(weights="DEFAULT", transform_input=False).to(device).eval()
    preds = get_inception_preds(model, images, device, batch_size)

    scores = []
    split_size = len(preds) // splits
    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


# ===================== CLI =====================

def parse_args():
    p = argparse.ArgumentParser(description="Text-to-image watermark quality metrics")
    p.add_argument("--wm_dir", type=str, required=True, help="watermarked image directory")
    p.add_argument("--real_dir", type=str, default="./output_generate_orig",
                   help="reference real/clean images for FID")
    p.add_argument("--annotations", type=str,
                   default="../dataset/annotations/captions_val2017.json")
    p.add_argument("--image_dir", type=str, default="../dataset/val2017")
    p.add_argument("--metrics", choices=["clip", "fid", "is", "all"], default="all")
    p.add_argument("--img_num", type=int, default=None, help="limit CLIP pairs")
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] wm_dir={args.wm_dir}, metrics={args.metrics}, device={device}")

    results = {}
    if args.metrics in ("clip", "all"):
        mean_clip, _ = evaluate_clip_score(
            args.wm_dir, args.annotations, args.image_dir, args.img_num, device
        )
        results["CLIP Score"] = mean_clip
        print(f"Average CLIP Score: {mean_clip:.4f}")

    if args.metrics in ("fid", "all"):
        fid = calculate_fid(args.real_dir, args.wm_dir, device, args.batch_size)
        results["FID"] = fid
        print(f"FID: {fid:.4f}")

    if args.metrics in ("is", "all"):
        is_mean, is_std = calculate_inception_score(
            args.wm_dir, splits=args.is_splits, batch_size=args.batch_size, device=device
        )
        results["Inception Score"] = (is_mean, is_std)
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

    print("\n" + "=" * 50)
    print("T2I Quality Summary")
    print("=" * 50)
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"  {k}: {v[0]:.4f} ± {v[1]:.4f}")
        else:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
