import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import os
import io
import time
from tqdm import tqdm
import random
# from diffusers import FluxImg2ImgPipeline
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline

class SD3FlowTrajectoryWatermarker:
    def __init__(self, model_id, patch_size=128, strength=0.02, device="cuda", num_chars=4):
        # 设置随机种子以确保可复现
        torch.manual_seed(42)  # 设置CPU端的种子
        torch.cuda.manual_seed_all(42)  # 设置GPU端的种子
        np.random.seed(42)  # 设置NumPy的种子
        random.seed(42)  # 设置Python随机库的种子
        torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
        torch.backends.cudnn.benchmark = False  # 禁用cuDNN的优化，避免硬件平台不同的结果差异

        self.dtype = torch.float16
        self.device = device
        self.patch_size = patch_size
        self.strength = strength 
        self.num_chars = num_chars
        self.num_bits = num_chars * 8
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(device)
        
        self.bases = self._generate_circular_bases()

    def _generate_circular_bases(self):
        """生成真正具有旋转不变性的径向对称基阵"""
        bases = []
        lin = torch.linspace(-1, 1, self.patch_size, device=self.device, dtype=self.dtype)
        y, x = torch.meshgrid(lin, lin, indexing='ij')

        rho = torch.sqrt(x**2 + y**2)

        for i in range(self.num_bits):
            torch.manual_seed(1024 + i)
            freq = 5.0 + (i * 4.0)  # 线性增加频率，确保每个 bit 占据不同的频带
            phase = torch.rand(1, device=self.device).item() * 2 * torch.pi

            circular_pattern = torch.sin(freq * torch.pi * rho + phase)

            mask = torch.exp(-(rho**2) / 0.8)
            circular_base = circular_pattern * mask

            base = circular_base.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)
            base = (base - base.mean()) / (base.std() + 1e-5)

            bases.append(base)
        return bases

    def _msg_to_bits(self, msg):
        msg = msg[:self.num_chars].ljust(self.num_chars)
        return "".join([bin(ord(c))[2:].zfill(8) for c in msg])

    def _bits_to_msg(self, bits):
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            try: chars.append(chr(int(byte, 2)))
            except: chars.append('?')
        return "".join(chars).strip()

    @torch.no_grad()
    def embed(self, input_image, message, denoising_strength=0.3):
        bits = self._msg_to_bits(message)
        v_modifier = torch.zeros_like(self.bases[0])
        for i, bit in enumerate(bits):
            if bit == '1': v_modifier += self.bases[i]
        
        # input_image 支持路径或 PIL 对象
        init_img = input_image if isinstance(input_image, Image.Image) else Image.open(input_image)
        init_img = init_img.convert("RGB").resize((1024, 1024))

        def trajectory_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            norm_t = t_val / 1000.0
            weight = 4 * norm_t * (1 - norm_t) 
            if 100 < t_val < 700:
                _, _, h, w = latents.shape
                tiled_v = v_modifier.repeat(1, 1, h//self.patch_size + 1, w//self.patch_size + 1)[:, :, :h, :w]
                latents += self.strength * weight * tiled_v
            kwargs["latents"] = latents
            return kwargs

        output_image = self.pipe(
            prompt="", image=init_img, strength=denoising_strength,
            num_inference_steps=28, guidance_scale=2.0, callback_on_step_end=trajectory_callback
        ).images[0]
        return output_image

    @torch.no_grad()
    def extract(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)
        
        decoded_bits = ""
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.view(-1), self.bases[i].view(-1), dim=0)
            decoded_bits += "1" if sim.item() > 0.008 else "0"
        return self._bits_to_msg(decoded_bits), decoded_bits
    
    @torch.no_grad()
    def detect(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)

        total_sim = 0
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.reshape(-1), self.bases[i].reshape(-1), dim=0)
            total_sim += abs(sim.item())
        
        score = total_sim / self.num_bits
        is_detected = score > 0.005
        return {
            "is_watermarked": is_detected,
            "confidence_score": round(score, 6),
        }


class SD3Text2ImgWatermarker:
    def __init__(self, model_id, patch_size=128, strength=0.02, device="cuda", num_chars=4):
        # 设置随机种子以确保可复现
        torch.manual_seed(42)  # 设置CPU端的种子
        torch.cuda.manual_seed_all(42)  # 设置GPU端的种子
        np.random.seed(42)  # 设置NumPy的种子
        random.seed(42)  # 设置Python随机库的种子
        torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
        torch.backends.cudnn.benchmark = False  # 禁用cuDNN的优化，避免硬件平台不同的结果差异

        self.dtype = torch.float16
        self.device = device
        self.patch_size = patch_size
        self.strength = strength 
        self.num_chars = num_chars
        self.num_bits = num_chars * 8
        self.pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=self.dtype).to(device)
        
        self.bases = self._generate_circular_bases()

    def _generate_circular_bases(self):
        """生成真正具有旋转不变性的径向对称基阵"""
        bases = []
        lin = torch.linspace(-1, 1, self.patch_size, device=self.device, dtype=self.dtype)
        y, x = torch.meshgrid(lin, lin, indexing='ij')

        rho = torch.sqrt(x**2 + y**2)

        for i in range(self.num_bits):
            torch.manual_seed(1024 + i)
            freq = 5.0 + (i * 4.0)  # 线性增加频率，确保每个 bit 占据不同的频带
            phase = torch.rand(1, device=self.device).item() * 2 * torch.pi

            circular_pattern = torch.sin(freq * torch.pi * rho + phase)

            mask = torch.exp(-(rho**2) / 0.8)
            circular_base = circular_pattern * mask

            base = circular_base.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)
            base = (base - base.mean()) / (base.std() + 1e-5)

            bases.append(base)
        return bases

    def _msg_to_bits(self, msg):
        msg = msg[:self.num_chars].ljust(self.num_chars)
        return "".join([bin(ord(c))[2:].zfill(8) for c in msg])

    def _bits_to_msg(self, bits):
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            try: chars.append(chr(int(byte, 2)))
            except: chars.append('?')
        return "".join(chars).strip()
    
    @torch.no_grad()
    def original_generate(self, prompt, message):
        output_image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        return output_image

    @torch.no_grad()
    def embed(self, prompt, message, denoising_strength=0.3):
        bits = self._msg_to_bits(message)
        v_modifier = torch.zeros_like(self.bases[0])
        for i, bit in enumerate(bits):
            if bit == '1': v_modifier += self.bases[i]

        def trajectory_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            t_val = t.item() if isinstance(t, torch.Tensor) else t

            
            if 200 < t_val < 600:
                _, _, h, w = latents.shape
                tiled_v = v_modifier.repeat(1, 1, h//self.patch_size + 1, w//self.patch_size + 1)[:, :, :h, :w]
                latents += self.strength * tiled_v
            
            kwargs["latents"] = latents
            return kwargs

        # 文生图生成
        output_image = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
            callback_on_step_end=trajectory_callback
        ).images[0]
        
        return output_image

    @torch.no_grad()
    def extract(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)
        
        decoded_bits = ""
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.view(-1), self.bases[i].view(-1), dim=0)
            decoded_bits += "1" if sim.item() > 0.01 else "0"
        return self._bits_to_msg(decoded_bits), decoded_bits

    @torch.no_grad()
    def detect(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)

        total_sim = 0
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.reshape(-1), self.bases[i].reshape(-1), dim=0)
            total_sim += abs(sim.item())
        
        score = total_sim / self.num_bits
        is_detected = score > 0.002
        return {
            "is_watermarked": is_detected,
            "confidence_score": round(score, 6),
        }


# 图生图多字符 速度场推移 图像编辑
class SD3ImgEditWatermarker:
    def __init__(self, model_id, patch_size=128, strength=0.02, device="cuda", num_chars=4):
        print(f"[*] 初始化 SD3 轨迹扰动模型: {model_id}")
        # 设置随机种子以确保可复现
        torch.manual_seed(42)  # 设置CPU端的种子
        torch.cuda.manual_seed_all(42)  # 设置GPU端的种子
        np.random.seed(42)  # 设置NumPy的种子
        random.seed(42)  # 设置Python随机库的种子
        torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
        torch.backends.cudnn.benchmark = False  # 禁用cuDNN的优化，避免硬件平台不同的结果差异

        self.dtype = torch.float16
        self.device = device
        self.patch_size = patch_size
        self.strength = strength 
        self.num_chars = num_chars
        self.num_bits = num_chars * 8
        
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(device)
        
        self.bases = self._generate_circular_bases()

    def _generate_circular_bases(self):
        """生成真正具有旋转不变性的径向对称基阵"""
        bases = []
        lin = torch.linspace(-1, 1, self.patch_size, device=self.device, dtype=self.dtype)
        y, x = torch.meshgrid(lin, lin, indexing='ij')

        rho = torch.sqrt(x**2 + y**2)

        for i in range(self.num_bits):
            torch.manual_seed(1024 + i)
            freq = 5.0 + (i * 4.0)  # 线性增加频率，确保每个 bit 占据不同的频带
            phase = torch.rand(1, device=self.device).item() * 2 * torch.pi

            circular_pattern = torch.sin(freq * torch.pi * rho + phase)

            mask = torch.exp(-(rho**2) / 0.8)
            circular_base = circular_pattern * mask

            base = circular_base.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)
            base = (base - base.mean()) / (base.std() + 1e-5)

            bases.append(base)
        return bases
        

    def _msg_to_bits(self, msg):
        msg = msg[:self.num_chars].ljust(self.num_chars)
        return "".join([bin(ord(c))[2:].zfill(8) for c in msg])

    def _bits_to_msg(self, bits):
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            try: chars.append(chr(int(byte, 2)))
            except: chars.append('?')
        return "".join(chars).strip()

    @torch.no_grad()
    def embed(self, input_image_path, message, prompt, denoising_strength=0.5):
        bits = self._msg_to_bits(message)
        v_modifier = torch.zeros_like(self.bases[0])
        for i, bit in enumerate(bits):
            if bit == '1': v_modifier += self.bases[i]
        
        init_img = Image.open(input_image_path).convert("RGB").resize((1024, 1024))

        def trajectory_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            t_val = t.item() if isinstance(t, torch.Tensor) else t
            
            if 100 < t_val < 700:
                _, _, h, w = latents.shape
                tiled_v = v_modifier.repeat(1, 1, h//self.patch_size + 1, w//self.patch_size + 1)[:, :, :h, :w]
                latents += self.strength * tiled_v
            
            kwargs["latents"] = latents
            return kwargs

        # 使用传入的prompt进行内容编辑
        output_image = self.pipe(
            prompt=prompt,  # 使用传入的prompt
            image=init_img, 
            strength=denoising_strength,  # 提高去噪强度以允许更多内容变化
            num_inference_steps=28, 
            guidance_scale=7.0,  # 提高guidance_scale以更好地遵循prompt
            callback_on_step_end=trajectory_callback
        ).images[0]
        
        return output_image

    @torch.no_grad()
    def extract(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)
        
        decoded_bits = ""
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.view(-1), self.bases[i].view(-1), dim=0)
            decoded_bits += "1" if sim.item() > 0.008 else "0"
        return self._bits_to_msg(decoded_bits), decoded_bits

    @torch.no_grad()
    def detect(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((1024, 1024))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
        img_t = (img_t / 127.5) - 1.0
        latents = self.pipe.vae.encode(img_t).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg_block = residue.mean(dim=0, keepdim=True)
        avg_block = (avg_block - avg_block.mean()) / (avg_block.std() + 1e-5)

        total_sim = 0
        for i in range(self.num_bits):
            sim = F.cosine_similarity(avg_block.reshape(-1), self.bases[i].reshape(-1), dim=0)
            total_sim += abs(sim.item())
        
        score = total_sim / self.num_bits
        is_detected = score > 0.005
        return {
            "is_watermarked": is_detected,
            "confidence_score": round(score, 6),
        }
