"""
文生图水印批处理（合并自 draft/text_to_image{,2,3}.py）

评测模式:
  --metric bit|detect|both
攻击类型:
  --attacks traditional|vae|all
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import os

import torch
from tqdm import tqdm

from watermarker import SD3Text2ImgWatermarker
from utils import (
    TRADITIONAL_ATTACKS,
    bit_accuracy,
    build_vae_networks,
    load_clean_image,
    load_coco_prompts,
    report_bit_accuracy,
    report_detection,
    run_traditional_attack,
    vae_compress,
)


def batch_process(
    output_dir,
    model_path,
    img_num,
    message="SDFLOW",
    strength=0.005,
    denoising_strength=0.06,
    metric="both",
    attacks="traditional",
    clean_dir="./output_generate_orig",
    vae_qualities=(1, 2, 3, 4, 5, 6),
    gpu_ids="0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.makedirs(output_dir, exist_ok=True)

    do_bit = metric in ("bit", "both")
    do_detect = metric in ("detect", "both")
    do_trad = attacks in ("traditional", "all")
    do_vae = attacks in ("vae", "all")

    marker = SD3Text2ImgWatermarker(model_path, strength=strength, num_chars=len(message))
    target_bits = marker._msg_to_bits(message) if do_bit else None

    labels, prompts = load_coco_prompts()
    img_num = min(img_num, len(prompts))
    print(f"[*] prompts={len(prompts)}, run={img_num}, metric={metric}, attacks={attacks}")

    device = None
    vae_nets = None
    if do_vae:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[*] Loading VAE attack networks...")
        vae_nets = build_vae_networks(device, qualities=vae_qualities)

    bit_stats = {}
    det_stats = {"score_clean": [], "score_wm": [], "score_attacks": {}, "det_acc": {}}

    def record_bit(name, img):
        _, bits = marker.extract(img)
        bit_stats.setdefault(name, []).append(bit_accuracy(target_bits, bits))

    def record_det(name, img):
        result = marker.detect(img)
        det_stats["score_attacks"].setdefault(name, []).append(result["confidence_score"])
        det_stats["det_acc"].setdefault(name, []).append(1 if result["is_watermarked"] else 0)

    for i in tqdm(range(img_num)):
        wm = marker.embed(prompts[i], message, denoising_strength=denoising_strength)
        wm.save(os.path.join(output_dir, f"{labels[i]}_w.png"))

        if do_bit:
            record_bit("none", wm)

        if do_detect:
            clean = load_clean_image(labels[i], clean_dir)
            if clean is not None:
                det_stats["score_clean"].append(marker.detect(clean)["confidence_score"])
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
                attacked = vae_compress(wm, net, device, output_size=wm.size)
                if do_bit:
                    record_bit(name, attacked)
                if do_detect:
                    record_det(name, attacked)

    if do_bit:
        report_bit_accuracy(bit_stats, img_num)
    if do_detect:
        report_detection(det_stats)


def parse_args():
    p = argparse.ArgumentParser(description="SD3 text-to-image watermark evaluation")
    p.add_argument("--output_dir", default="./output_generate")
    p.add_argument("--model_path", default="../../llm_model/stable-diffusion-3-medium-diffusers")
    p.add_argument("--img_num", type=int, default=100)
    p.add_argument("--message", default="SDFLOW")
    p.add_argument("--strength", type=float, default=0.005)
    p.add_argument("--denoising_strength", type=float, default=0.06)
    p.add_argument("--metric", choices=["bit", "detect", "both"], default="both")
    p.add_argument("--attacks", choices=["traditional", "vae", "all"], default="traditional")
    p.add_argument("--clean_dir", default="./output_generate_orig")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--vae_qualities", default="1,2,3,4,5,6")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qualities = tuple(int(x) for x in args.vae_qualities.split(",") if x.strip())
    batch_process(
        output_dir=args.output_dir,
        model_path=args.model_path,
        img_num=args.img_num,
        message=args.message,
        strength=args.strength,
        denoising_strength=args.denoising_strength,
        metric=args.metric,
        attacks=args.attacks,
        clean_dir=args.clean_dir,
        vae_qualities=qualities,
        gpu_ids=args.gpu_ids,
    )
