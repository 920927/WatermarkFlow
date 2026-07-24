"""
图生图场景水印图像质量度量（PSNR / MSE / SSIM / LPIPS）

与文生图质量脚本 t2i_quality.py 对应。
"""
import torch
import torch.nn.functional as F
import math
from PIL import Image
import argparse
import random
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, Optional
import glob

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """
    计算两幅图像之间的PSNR（峰值信噪比）
    """
    if img1.shape != img2.shape:
        # 尝试调整大小使其匹配
        if len(img1.shape) == 3 and len(img2.shape) == 3:
            # 确保都是 [C, H, W] 格式
            if img1.shape[0] != img2.shape[0]:
                raise ValueError(f"Channel dimension mismatch: {img1.shape[0]} vs {img2.shape[0]}")
            
            # 调整较小的图像到较大图像的尺寸
            h1, w1 = img1.shape[1], img1.shape[2]
            h2, w2 = img2.shape[1], img2.shape[2]
            
            if h1 != h2 or w1 != w2:
                print(f"Warning: Resizing images to match: {img1.shape} -> {img2.shape}")
                if h1 * w1 > h2 * w2:
                    # img1较大，调整img2
                    img2 = F.interpolate(img2.unsqueeze(0), size=(h1, w1), mode='bilinear', align_corners=False).squeeze(0)
                else:
                    # img2较大，调整img1
                    img1 = F.interpolate(img1.unsqueeze(0), size=(h2, w2), mode='bilinear', align_corners=False).squeeze(0)
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(data_range) - 10 * math.log10(mse)
    return psnr


def calculate_image_similarity_metrics(img1: torch.Tensor, img2: torch.Tensor) -> dict:
    """
    计算多种图像相似度指标
    """
    metrics = {}
    metrics['psnr'] = calculate_psnr(img1, img2, data_range=1.0)
    metrics['mse'] = F.mse_loss(img1, img2).item()
    
    from torchmetrics.functional import structural_similarity_index_measure
    metrics['ssim'] = structural_similarity_index_measure(img1.unsqueeze(0), img2.unsqueeze(0)).item()
    
    import lpips
    lpips_model = lpips.LPIPS(net='alex', verbose=False)
    metrics['lpips'] = lpips_model(img1.unsqueeze(0), img2.unsqueeze(0)).item()
    
    return metrics


def preprocess_image_for_psnr(original_image: Image.Image, processed_image: Image.Image, 
                             target_size: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预处理图像用于PSNR计算，确保尺寸匹配
    
    Args:
        original_image: 原始PIL图像
        processed_image: 处理后的PIL图像
        target_size: 目标尺寸 (宽, 高)，如果为None则使用原始图像尺寸
    
    Returns:
        tuple: (original_tensor, processed_tensor) 预处理后的张量
    """
    # 如果指定了目标尺寸，调整大小
    if target_size is not None:
        original_resized = original_image.resize(target_size, Image.LANCZOS)
        processed_resized = processed_image.resize(target_size, Image.LANCZOS)
    else:
        # 使用相同的尺寸
        target_size = original_image.size
        processed_resized = processed_image.resize(target_size, Image.LANCZOS)
        original_resized = original_image
    
    # 转换为numpy数组然后到torch张量
    orig_array = np.array(original_resized).astype(np.float32) / 255.0
    proc_array = np.array(processed_resized).astype(np.float32) / 255.0
    
    # 转换为 [C, H, W] 格式
    if len(orig_array.shape) == 3 and orig_array.shape[-1] == 3:  # [H, W, C]
        orig_array = orig_array.transpose(2, 0, 1)
    if len(proc_array.shape) == 3 and proc_array.shape[-1] == 3:  # [H, W, C]
        proc_array = proc_array.transpose(2, 0, 1)
    
    # 转换为torch张量
    original_tensor = torch.from_numpy(orig_array)
    processed_tensor = torch.from_numpy(proc_array)
    
    return original_tensor, processed_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-to-image watermark quality metrics (PSNR/SSIM/LPIPS)")
    parser.add_argument("--orig_dir", type=str, default="../dataset/val2017", help="original image directory")
    parser.add_argument("--watermarked_dir", type=str, default="./output_track", help="watermarked image directory")
    parser.add_argument("--image_extensions", type=str, default="jpg,png,jpeg,bmp,tiff", help="comma-separated image extensions")
    parser.add_argument("--calculate_metrics", action="store_true", help="calculate detailed metrics")
    parser.add_argument("--no_watermark", action="store_true", help="run without watermark for baseline PSNR")
    parser.add_argument("--seed", default=4, help="run without watermark for baseline PSNR")
    parser.add_argument("--psnr_target_size", default="256x256", help="image size for psnr computation")
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")


    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    # 获取所有图片文件
    extensions = args.image_extensions.split(',')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(args.watermarked_dir, f"*.{ext}")))
        image_files.extend(glob.glob(os.path.join(args.watermarked_dir, f"*.{ext.upper()}")))

    imgs_psnr, imgs_mse, imgs_ssim, imgs_lpips = [], [], [], []
    for img_idx, img_path in enumerate(tqdm(image_files)):

        watermarked_image = Image.open(img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        orig_w, orig_h = watermarked_image.size

        psnr_target_size = args.psnr_target_size
        if psnr_target_size == "original":
            target_size = (orig_w, orig_h)
        else:
            try:
                w, h = map(int, psnr_target_size.split('x'))
                target_size = (w, h)
            except:
                target_size = (orig_w, orig_h)  

        original_img_path = os.path.join(args.orig_dir, f"{img_name}.jpg")
        original_image = Image.open(original_img_path).convert('RGB')
        original_tensor, watermarked_tensor = preprocess_image_for_psnr(
            original_image, 
            watermarked_image,
            target_size=target_size
        )
    
        img_psnr = calculate_psnr(original_tensor, watermarked_tensor, data_range=1.0)
        imgs_psnr.append(img_psnr)

        similarity_metrics = calculate_image_similarity_metrics(original_tensor, watermarked_tensor)
        imgs_mse.append(similarity_metrics['mse'])
        imgs_ssim.append(similarity_metrics['ssim'])
        imgs_lpips.append(similarity_metrics['lpips'])


    print('Average image psnr:', sum(imgs_psnr)/len(imgs_psnr))
    print('Average image mse:', sum(imgs_mse)/len(imgs_mse))
    print('Average image ssim:', sum(imgs_ssim)/len(imgs_ssim))
    print('Average image lpips:', sum(imgs_lpips)/len(imgs_lpips))
