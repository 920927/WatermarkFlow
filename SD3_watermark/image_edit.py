"""
图像编辑水印批处理（合并自 draft/image_edit{,2,3}.py）

评测模式:
  --metric bit     : bit 准确率（原 image_edit2.py）
  --metric detect  : 检测分数 + ROC（原 image_edit3.py）
  --metric both    : 两者都做

攻击类型:
  --attacks traditional|vae|all

单张 demo（对应原 image_edit.py）:
  python image_edit.py --demo --img_in 000000001584.jpg --edit_prompt "..."
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

from watermarker import SD3ImgEditWatermarker
from utils import (
    TRADITIONAL_ATTACKS,
    bit_accuracy,
    build_vae_networks,
    load_wbench_edits,
    report_bit_accuracy,
    report_detection,
    run_traditional_attack,
    vae_compress,
)


def run_demo(model_path, img_in, message, edit_prompt, strength, denoising_strength, save_path):
    marker = SD3ImgEditWatermarker(model_path, strength=strength, num_chars=len(message))
    print("\n[demo] 编辑嵌入水印...")
    wm = marker.embed(img_in, message, prompt=edit_prompt, denoising_strength=denoising_strength)
    wm.save(save_path)
    print(f"[demo] saved -> {save_path}")


def batch_process(
    output_dir,
    model_path,
    img_num=100,
    message="SDFLOW",
    strength=0.003,
    denoising_strength=0.8,
    metric="both",
    attacks="traditional",
    prompts_csv="../../llm_model/W-Bench/DET_INVERSION_1K/prompts.csv",
    image_dir="../../llm_model/W-Bench/DET_INVERSION_1K/image",
    vae_qualities=(1, 2, 3, 4, 5, 6),
    gpu_ids="0",
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.makedirs(output_dir, exist_ok=True)

    do_bit = metric in ("bit", "both")
    do_detect = metric in ("detect", "both")
    do_trad = attacks in ("traditional", "all")
    do_vae = attacks in ("vae", "all")

    marker = SD3ImgEditWatermarker(model_path, strength=strength, num_chars=len(message))
    target_bits = marker._msg_to_bits(message) if do_bit else None

    samples = load_wbench_edits(prompts_csv, image_dir, limit=img_num)
    print(f"[*] samples={len(samples)}, metric={metric}, attacks={attacks}")

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

    for img_path, edit_prompt, stem in tqdm(samples):
        orig = Image.open(img_path).convert("RGB")

        wm = marker.embed(
            img_path, message, prompt=edit_prompt, denoising_strength=denoising_strength
        )
        wm = wm.resize(orig.size, Image.Resampling.LANCZOS)
        wm.save(os.path.join(output_dir, f"{stem}.png"))

        if do_bit:
            record_bit("none", wm)

        if do_detect:
            det_stats["score_clean"].append(marker.detect(orig)["confidence_score"])
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

    if do_bit:
        report_bit_accuracy(bit_stats, len(samples))
    if do_detect:
        report_detection(det_stats)


def parse_args():
    p = argparse.ArgumentParser(description="SD3 image-edit watermark evaluation")
    p.add_argument("--demo", action="store_true", help="单张编辑 demo（原 image_edit.py）")
    p.add_argument("--img_in", default="000000001584.jpg")
    p.add_argument("--edit_prompt", default="a fire engine on the road, high quality, detailed, photorealistic")
    p.add_argument("--demo_out", default="edit3.png")

    p.add_argument("--output_dir", default="./output_edit")
    p.add_argument("--model_path", default="../../llm_model/stable-diffusion-3-medium-diffusers")
    p.add_argument("--img_num", type=int, default=100)
    p.add_argument("--message", default="SDFLOW")
    p.add_argument("--strength", type=float, default=0.003)
    p.add_argument("--denoising_strength", type=float, default=0.8)
    p.add_argument("--metric", choices=["bit", "detect", "both"], default="both")
    p.add_argument("--attacks", choices=["traditional", "vae", "all"], default="traditional")
    p.add_argument("--prompts_csv", default="../../llm_model/W-Bench/DET_INVERSION_1K/prompts.csv")
    p.add_argument("--image_dir", default="../../llm_model/W-Bench/DET_INVERSION_1K/image")
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--vae_qualities", default="1,2,3,4,5,6")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        run_demo(
            args.model_path, args.img_in, args.message, args.edit_prompt,
            args.strength, args.denoising_strength, args.demo_out,
        )
    else:
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
            prompts_csv=args.prompts_csv,
            image_dir=args.image_dir,
            vae_qualities=qualities,
            gpu_ids=args.gpu_ids,
        )
