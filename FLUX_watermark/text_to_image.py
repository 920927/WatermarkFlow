"""
FLUX 文生图水印批处理

评测模式:
  --metric bit|detect|both
攻击类型:
  --attacks traditional|vae|all
"""
import argparse
import glob
import os
import sys
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from tqdm import tqdm

from watermarker import FluxText2ImgWatermarker
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
    extract_threshold=0.01,
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

    marker = FluxText2ImgWatermarker(
        model_path,
        strength=strength,
        num_chars=len(message),
        extract_threshold=extract_threshold,
    )
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
            det_stats["det_acc"].setdefault("none", []).append(
                1 if result["is_watermarked"] else 0
            )

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


def eval_existing(
    wm_dir,
    clean_dir,
    model_path,
    img_num,
    message="SDFLOW",
    strength=0.005,
    extract_threshold=0.01,
    metric="both",
    attacks="traditional",
    vae_qualities=(1, 2, 3, 4, 5, 6),
    gpu_ids="0",
):
    """对已生成的水印图做鲁棒性评测（不再重新生成）。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    do_bit = metric in ("bit", "both")
    do_detect = metric in ("detect", "both")
    do_trad = attacks in ("traditional", "all")
    do_vae = attacks in ("vae", "all")

    wm_files = sorted(glob.glob(os.path.join(wm_dir, "*_w.png")))
    if img_num is not None:
        wm_files = wm_files[:img_num]
    if not wm_files:
        raise RuntimeError(f"No watermarked images found in {wm_dir}")

    marker = FluxText2ImgWatermarker(
        model_path, strength=strength, num_chars=len(message),
        extract_threshold=extract_threshold,
    )
    target_bits = marker._msg_to_bits(message) if do_bit else None

    device = None
    vae_nets = None
    if do_vae:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def label_from_path(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.replace("_w", ""))

    print(f"[*] eval_only wm_dir={wm_dir}, n={len(wm_files)}, metric={metric}, attacks={attacks}")

    for path in tqdm(wm_files):
        wm = Image.open(path).convert("RGB")
        lab = label_from_path(path)

        if do_bit:
            record_bit("none", wm)

        if do_detect:
            clean = load_clean_image(lab, clean_dir)
            if clean is not None:
                det_stats["score_clean"].append(marker.detect(clean)["confidence_score"])
            result = marker.detect(wm)
            det_stats["score_wm"].append(result["confidence_score"])
            det_stats["det_acc"].setdefault("none", []).append(
                1 if result["is_watermarked"] else 0
            )

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
        report_bit_accuracy(bit_stats, len(wm_files))
    if do_detect:
        report_detection(det_stats)


def parse_args():
    p = argparse.ArgumentParser(description="FLUX text-to-image watermark evaluation")
    p.add_argument("--output_dir", default="./output_flux")
    p.add_argument("--model_path", default="../../llm_model/FLUX.1-dev")
    p.add_argument("--img_num", type=int, default=100)
    p.add_argument("--message", default="SDFLOW")
    p.add_argument("--strength", type=float, default=0.005)
    p.add_argument("--denoising_strength", type=float, default=0.06)
    p.add_argument("--extract_threshold", type=float, default=0.01)
    p.add_argument("--metric", choices=["bit", "detect", "both"], default="both")
    p.add_argument("--attacks", choices=["traditional", "vae", "all"], default="traditional")
    p.add_argument("--clean_dir", default="./output_generate_orig")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--vae_qualities", default="1,2,3,4,5,6")
    p.add_argument("--eval_only", action="store_true", help="evaluate existing wm images only")
    p.add_argument("--wm_dir", default=None, help="wm dir for --eval_only (default: --output_dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    qualities = tuple(int(x) for x in args.vae_qualities.split(",") if x.strip())
    if args.eval_only:
        eval_existing(
            wm_dir=args.wm_dir or args.output_dir,
            clean_dir=args.clean_dir,
            model_path=args.model_path,
            img_num=args.img_num,
            message=args.message,
            strength=args.strength,
            extract_threshold=args.extract_threshold,
            metric=args.metric,
            attacks=args.attacks,
            vae_qualities=qualities,
            gpu_ids=args.gpu_ids,
        )
    else:
        batch_process(
            output_dir=args.output_dir,
            model_path=args.model_path,
            img_num=args.img_num,
            message=args.message,
            strength=args.strength,
            denoising_strength=args.denoising_strength,
            extract_threshold=args.extract_threshold,
            metric=args.metric,
            attacks=args.attacks,
            clean_dir=args.clean_dir,
            vae_qualities=qualities,
            gpu_ids=args.gpu_ids,
        )
