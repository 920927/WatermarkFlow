import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import os
import io
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
import os
import time
from tqdm import tqdm
from watermarker import SD3Text2ImgWatermarker
from utils import calculate_psnr, apply_attack
import glob
import json
from watermarker import SD3ImgEditWatermarker



# ================= 运行测试 =================


if __name__ == "__main__":
    MODEL = "/ruisun2025/dx/llm_model/stable-diffusion-3-medium-diffusers"
    IMG_IN = "000000000285.jpg"

    TEST_MSG = "SDFLOW"
    
    # 示例：将熊改为猫的prompt
    EDIT_PROMPT = "a cute cat in the forest, high quality, detailed, photorealistic"

    marker = SD3ImgEditWatermarker(MODEL, strength=0.003, num_chars=len(TEST_MSG))
    
    print("\n[1] 开始嵌入轨迹扰动（内容编辑）...")
    # 提高denoising_strength以允许更多内容变化
    wm_img = marker.embed(IMG_IN, TEST_MSG, 
                          prompt=EDIT_PROMPT,  # 传入编辑prompt
                          denoising_strength=0.8)  # 适当提高强度
    wm_img.save("edit3.png")

