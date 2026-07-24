import os
import argparse
import json
import shutil
import tqdm
from pathlib import Path
from PIL import Image, ImageFilter
import glob
import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Subset

from scipy.stats import binom
import matplotlib.pyplot as plt
from sklearn import metrics

import cv2
from watermarker import SD3FlowTrajectoryWatermarker
from utils import calculate_psnr, apply_attack

gpu_ids = "2" 
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

# ------------------------------------------------------------------------------------------------------------------------------------

def robustness_test(watermarked_dir, model_path, message):
    
    marker = SD3FlowTrajectoryWatermarker(model_path, strength=0.03, num_chars=len(message))
    watermarked_bits = marker._msg_to_bits(message)
    
    image_files = [f for f in os.listdir(watermarked_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"[*] 找到 {len(image_files)} 张图片，开始处理...")

    stats = {
        "acc_none": [],
        "acc_jpeg": [],
        "acc_blur": [],
        "acc_resize": []
    }

    for image_file in tqdm(image_files):
        
        image_path = os.path.join(watermarked_dir, image_file)
        wm_img = Image.open(image_path)

        _, bits_none = marker.extract(wm_img)
        stats["acc_none"].append(sum(b1==b2 for b1, b2 in zip(watermarked_bits, bits_none)) / len(watermarked_bits))
        
        # JPEG Q50 攻击
        img_jpeg = apply_attack(wm_img, "jpeg", 50)
        _, bits_jpeg = marker.extract(img_jpeg)
        stats["acc_jpeg"].append(sum(b1==b2 for b1, b2 in zip(watermarked_bits, bits_jpeg)) / len(watermarked_bits))
        
        # 高斯模糊 R=2 攻击
        img_blur = apply_attack(wm_img, "blur", 2)
        _, bits_blur = marker.extract(img_blur)
        stats["acc_blur"].append(sum(b1==b2 for b1, b2 in zip(watermarked_bits, bits_blur)) / len(watermarked_bits))

        # 缩放 0.5x 攻击
        img_res = apply_attack(wm_img, "resize", 0)
        _, bits_res = marker.extract(img_res)
        stats["acc_resize"].append(sum(b1==b2 for b1, b2 in zip(watermarked_bits, bits_res)) / len(watermarked_bits))

    # --- 输出统计报告 ---
    print("\n" + "="*50)
    print(f"批处理统计报告 (样本数: {len(image_files)})")
    print("-" * 50)
    print(f"平均准确率 (无攻击):  {np.mean(stats['acc_none'])*100:.1f}%")
    print(f"平均准确率 (JPEG Q50):{np.mean(stats['acc_jpeg'])*100:.1f}%")
    print(f"平均准确率 (高斯模糊): {np.mean(stats['acc_blur'])*100:.1f}%")
    print(f"平均准确率 (缩放攻击): {np.mean(stats['acc_resize'])*100:.1f}%")
    print("="*50)
    
    

if __name__ == "__main__":
    WATERMARK_MESSAGE = "SDFLOW"
    WATERMARKED_FOLDER = "./output_track"
    MODEL_WEIGHTS = "../../llm_model/stable-diffusion-3-medium-diffusers"
    
    robustness_test(WATERMARKED_FOLDER, MODEL_WEIGHTS, message=WATERMARK_MESSAGE)