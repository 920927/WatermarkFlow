import numpy as np
import os
import io
from PIL import Image, ImageFilter, ImageEnhance
import random


def attack_random_mask(image, param=0.3):
    # 获取图片尺寸
    w, h = image.size
    
    # 计算涂黑区域的最大尺寸
    max_area = w * h
    mask_area = int(max_area * param)  # 涂黑区域的面积
    
    # 随机确定涂黑区域的高度和宽度
    # 确保宽高比在合理范围内（0.5到2之间）
    aspect_ratio = random.uniform(0.5, 2.0)
    mask_height = int((mask_area / aspect_ratio) ** 0.5)
    mask_width = int(mask_height * aspect_ratio)
    
    # 确保不超过图片边界
    mask_width = min(mask_width, w)
    mask_height = min(mask_height, h)
    
    if mask_width < 10 or mask_height < 10:  # 防止区域太小
        mask_width = min(w, 10)
        mask_height = min(h, 10)
    
    # 随机确定涂黑区域的左上角位置
    left = random.randint(0, w - mask_width)
    top = random.randint(0, h - mask_height)
    right = left + mask_width
    bottom = top + mask_height
    
    # 创建图片的副本
    masked_image = image.copy()
    
    # 创建黑色区域
    from PIL import ImageDraw
    draw = ImageDraw.Draw(masked_image)
    draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
    
    return masked_image

# --- 工具函数 ---

def calculate_psnr(img1, img2, img_size=1024):
    img1_arr = np.array(img1.convert("RGB").resize((img_size, img_size))).astype(np.float64)
    img2_arr = np.array(img2.convert("RGB").resize((img_size, img_size))).astype(np.float64)
    mse = np.mean((img1_arr - img2_arr) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 100.0

def apply_attack(image, attack_type, param):
    if attack_type == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=param)
        return Image.open(buf)
    
    elif attack_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=param))
    
    elif attack_type == "resize":
        w, h = image.size
        scale = param
        return image.resize((int(w*scale), int(h*scale))).resize((w, h), Image.Resampling.LANCZOS)
    
    elif attack_type == "crop":
        return attack_random_mask(image, param)
    
    elif attack_type == "rotate":
        return image.rotate(param)
    
    elif attack_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(param)
    
    elif attack_type == "gaussian_noise":
        image_array = np.array(image)
        noise = np.random.normal(0, param, image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))
    
    elif attack_type == "uniform_noise":
        image_array = np.array(image)
        noise = np.random.uniform(-param, param, image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))
    
    elif attack_type == "salt_pepper_noise":
        image_array = np.array(image)
        total_pixels = image_array.size
        salt_pepper_count = int(param * total_pixels)
        for _ in range(salt_pepper_count):
            x = random.randint(0, image_array.shape[0] - 1)
            y = random.randint(0, image_array.shape[1] - 1)
            if random.random() < 0.5:
                image_array[x, y] = 255  # salt
            else:
                image_array[x, y] = 0    # pepper
        return Image.fromarray(image_array)
    
    elif attack_type == "exponential_noise":
        image_array = np.array(image)
        noise = np.random.exponential(param, image_array.shape)
        noisy_image = np.clip(image_array + noise, 0, 255)
        return Image.fromarray(noisy_image.astype(np.uint8))
    
    elif attack_type == "poisson_noise":
        image_array = np.array(image)
        noisy_image = np.random.poisson(image_array / 255.0 * param) * 255
        return Image.fromarray(np.clip(noisy_image, 0, 255).astype(np.uint8))
    
    elif attack_type == "filter":
        return image.filter(ImageFilter.CONTOUR)  # or any other filter, e.g., EDGE_ENHANCE

    else:
        return image


