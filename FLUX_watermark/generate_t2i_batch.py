"""
FLUX 文生图批量生成：干净图（无水印）+ 水印图，支持断点续跑。

示例:
  python generate_t2i_batch.py --task both --img_num 1000 --gpu_ids 1
  python generate_t2i_batch.py --task clean --img_num 1000 --gpu_ids 0
  python generate_t2i_batch.py --task wm --img_num 1000 --gpu_ids 1
"""
import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from diffusers import FluxPipeline
from tqdm import tqdm

from utils import load_coco_prompts
from watermarker import FluxText2ImgWatermarker


def generate_clean(pipe, labels, prompts, out_dir, seed_base=42, skip_existing=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (lab, prompt) in enumerate(tqdm(zip(labels, prompts), total=len(labels), desc="clean")):
        path = out_dir / f"{lab}.png"
        if skip_existing and path.exists():
            continue
        gen = torch.Generator(device=pipe.device.type).manual_seed(seed_base + i)
        img = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=3.5,
            generator=gen,
        ).images[0]
        img.save(path)


def generate_wm(marker, labels, prompts, out_dir, message, denoising_strength, skip_existing=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for lab, prompt in tqdm(zip(labels, prompts), total=len(labels), desc="wm"):
        path = out_dir / f"{lab}_w.png"
        if skip_existing and path.exists():
            continue
        img = marker.embed(prompt, message, denoising_strength=denoising_strength)
        img.save(path)


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    labels, prompts = load_coco_prompts()
    n = min(args.img_num, len(prompts))
    labels, prompts = labels[:n], prompts[:n]
    print(f"[*] task={args.task}, n={n}, clean_dir={args.clean_dir}, wm_dir={args.wm_dir}")

    if args.task in ("clean", "both"):
        print("[*] Loading FLUX pipeline for clean generation...")
        pipe = FluxPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).to("cuda")
        generate_clean(pipe, labels, prompts, args.clean_dir, args.seed, args.skip_existing)
        del pipe
        torch.cuda.empty_cache()
        print(f"[*] Clean done -> {args.clean_dir}")

    if args.task in ("wm", "both"):
        print("[*] Loading watermarker for WM generation...")
        marker = FluxText2ImgWatermarker(
            args.model_path,
            strength=args.strength,
            num_chars=len(args.message),
            extract_threshold=args.extract_threshold,
        )
        generate_wm(
            marker, labels, prompts, args.wm_dir, args.message,
            args.denoising_strength, args.skip_existing,
        )
        print(f"[*] WM done -> {args.wm_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="FLUX T2I batch generation (clean + watermarked)")
    p.add_argument("--task", choices=["clean", "wm", "both"], default="both")
    p.add_argument("--clean_dir", default="./output_flux_t2i_1000_clean")
    p.add_argument("--wm_dir", default="./output_flux_t2i_1000")
    p.add_argument("--model_path", default="../../llm_model/FLUX.1-dev")
    p.add_argument("--img_num", type=int, default=1000)
    p.add_argument("--message", default="SDFLOW")
    p.add_argument("--strength", type=float, default=0.005)
    p.add_argument("--denoising_strength", type=float, default=0.06)
    p.add_argument("--extract_threshold", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_ids", default="0")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
