"""
图生图水印批处理（合并自 draft/image_to_image{,2,3,4}.py）

评测模式:
  --metric bit     : bit 准确率（原 image_to_image / image_to_image3）
  --metric detect  : 检测分数 + ROC（原 image_to_image2 / image_to_image4）
  --metric both    : 两者都做

攻击类型:
  --attacks traditional : 传统图像攻击
  --attacks vae         : CompressAI VAE 压缩攻击
  --attacks all         : 两者都做
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os

import torch
from PIL import Image
from tqdm import tqdm

from watermarker import SD3FlowTrajectoryWatermarker
from utils import (
    TRADITIONAL_ATTACKS,
    bit_accuracy,
    build_vae_networks,
    calculate_psnr,
    list_image_files,
    report_bit_accuracy,
    report_detection,
    report_psnr,
    run_traditional_attack,
    vae_compress,
)


def batch_process(
    input_dir,
    output_dir,
    model_path,
    img_num=None,
    message="SDFLOW",
    strength=0.06,
    denoising_strength=0.06,
    metric="both",
    attacks="traditional",
    vae_qualities=(1, 2, 3, 4, 5, 6),
    gpu_ids="0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.makedirs(output_dir, exist_ok=True)

    do_bit = metric in ("bit", "both")
    do_detect = metric in ("detect", "both")
    do_trad = attacks in ("traditional", "all")
    do_vae = attacks in ("vae", "all")

    marker = SD3FlowTrajectoryWatermarker(
        model_path, strength=strength, num_chars=len(message)
    )
    target_bits = marker._msg_to_bits(message) if do_bit else None

    image_files = list_image_files(input_dir, limit=img_num)
    print(f"[*] images={len(image_files)}, metric={metric}, attacks={attacks}")

    device = None
    vae_nets = None
    if do_vae:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[*] Loading VAE attack networks...")
        vae_nets = build_vae_networks(device, qualities=vae_qualities)

    psnr_list = []
    bit_stats = {}
    det_stats = {"score_clean": [], "score_wm": [], "score_attacks": {}, "det_acc": {}}

    def record_bit(name, img):
        _, bits = marker.extract(img)
        bit_stats.setdefault(name, []).append(bit_accuracy(target_bits, bits))

    def record_det(name, img):
        result = marker.detect(img)
        det_stats["score_attacks"].setdefault(name, []).append(result["confidence_score"])
        det_stats["det_acc"].setdefault(name, []).append(1 if result["is_watermarked"] else 0)

    for filename in tqdm(image_files):
        path = os.path.join(input_dir, filename)
        orig = Image.open(path).convert("RGB")
        save_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

        if do_detect:
            det_stats["score_clean"].append(marker.detect(orig)["confidence_score"])

        wm = marker.embed(path, message, denoising_strength=denoising_strength)
        wm_resized = wm.resize(orig.size, Image.Resampling.LANCZOS)
        wm_resized.save(save_path)

        psnr_list.append(calculate_psnr(orig, wm, img_size=256))

        if do_bit:
            record_bit("none", wm)

        if do_detect:
            result = marker.detect(wm)
            det_stats["score_wm"].append(result["confidence_score"])
            det_stats["det_acc"].setdefault("none", []).append(1 if result["is_watermarked"] else 0)

        if do_trad:
            for name, atk, param in TRADITIONAL_ATTACKS:
                _, attacked = run_traditional_attack(wm, name, atk, param)
                if do_bit:
                    record_bit(name, attacked)
                if do_detect:
                    record_det(name, attacked)

        if do_vae:
            for name, net in vae_nets.items():
                attacked = vae_compress(wm, net, device, output_size=orig.size)
                if do_bit:
                    record_bit(name, attacked)
                if do_detect:
                    record_det(name, attacked)

    report_psnr(psnr_list)
    if do_bit:
        report_bit_accuracy(bit_stats, len(image_files))
    if do_detect:
        report_detection(det_stats)


def parse_args():
    p = argparse.ArgumentParser(description="SD3 image-to-image watermark evaluation")
    p.add_argument("--input_dir", default="../dataset/val2017")
    p.add_argument("--output_dir", default="./output_track")
    p.add_argument("--model_path", default="../../llm_model/stable-diffusion-3-medium-diffusers")
    p.add_argument("--img_num", type=int, default=100)
    p.add_argument("--message", default="SDFLOW")
    p.add_argument("--strength", type=float, default=0.06)
    p.add_argument("--denoising_strength", type=float, default=0.06)
    p.add_argument("--metric", choices=["bit", "detect", "both"], default="both")
    p.add_argument("--attacks", choices=["traditional", "vae", "all"], default="traditional")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--vae_qualities", default="1,2,3,4,5,6")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qualities = tuple(int(x) for x in args.vae_qualities.split(",") if x.strip())
    batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        img_num=args.img_num,
        message=args.message,
        strength=args.strength,
        denoising_strength=args.denoising_strength,
        metric=args.metric,
        attacks=args.attacks,
        vae_qualities=qualities,
        gpu_ids=args.gpu_ids,
    )
