import io
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm
from diffusers import FluxPipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps

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
        is_detected = score > 0.002
        return {
            "is_watermarked": is_detected,
            "confidence_score": round(score, 6),
        }

# =============================================================================
# FLUX（整图单圆 base；img2img/edit 基于 FluxPipeline + strength 切片）
# =============================================================================

class _FluxWatermarkBase:
    """共享：单圆 base、VAE 编码、提取 / 检测、轨迹注入。"""

    def __init__(
        self,
        pipe,
        strength=0.005,
        device="cuda",
        num_chars=4,
        extract_threshold=0.01,
        height=1024,
        width=1024,
        patch_size=128,
    ):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.dtype = torch.bfloat16
        self.device = device
        self.patch_size = patch_size
        self.strength = strength
        self.extract_threshold = extract_threshold
        self.num_chars = num_chars
        self.num_bits = num_chars * 8
        self.height = height
        self.width = width
        self.pipe = pipe
        self.latent_h = height // 8
        self.latent_w = width // 8
        self.freqs, self.phases, self.bases = self._generate_circular_bases()

    def _radial_pattern(self, h, w, freq, phase):
        ly = torch.linspace(-1, 1, h, device=self.device, dtype=self.dtype)
        lx = torch.linspace(-1, 1, w, device=self.device, dtype=self.dtype)
        y, x = torch.meshgrid(ly, lx, indexing="ij")
        rho = torch.sqrt(x**2 + y**2)
        return torch.sin(freq * torch.pi * rho + phase) * torch.exp(-(rho**2) / 0.8)

    def _normalize_base(self, pattern):
        base = pattern.unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)
        return (base - base.mean()) / (base.std() + 1e-5)

    def _base_at(self, freq, phase, h, w):
        return self._normalize_base(self._radial_pattern(h, w, freq, phase))

    def _generate_circular_bases(self):
        freqs, phases, bases = [], [], []
        for i in range(self.num_bits):
            torch.manual_seed(1024 + i)
            freq = 5.0 + (i * 4.0)
            phase = torch.rand(1, device=self.device).item() * 2 * torch.pi
            freqs.append(freq)
            phases.append(phase)
            bases.append(self._base_at(freq, phase, self.latent_h, self.latent_w))
        return freqs, phases, bases

    def _modifier_at(self, h, w, bits):
        v = torch.zeros(1, 16, h, w, device=self.device, dtype=self.dtype)
        for i, bit in enumerate(bits):
            if bit == "1":
                v = v + self._base_at(self.freqs[i], self.phases[i], h, w)
        return v

    def _msg_to_bits(self, msg):
        msg = msg[: self.num_chars].ljust(self.num_chars)
        return "".join(bin(ord(c))[2:].zfill(8) for c in msg)

    def _bits_to_msg(self, bits):
        chars = []
        for i in range(0, len(bits), 8):
            try:
                chars.append(chr(int(bits[i : i + 8], 2)))
            except Exception:
                chars.append("?")
        return "".join(chars).strip()

    def _encode_latents(self, image_input):
        img = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
        img = img.convert("RGB").resize((self.width, self.height))
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        img_t = (img_t / 127.5) - 1.0
        z = self.pipe.vae.encode(img_t).latent_dist.mode()
        shift = getattr(self.pipe.vae.config, "shift_factor", 0.0) or 0.0
        return (z - shift) * self.pipe.vae.config.scaling_factor

    def _get_residue(self, image_input):
        latents = self._encode_latents(image_input)
        residue = latents - F.avg_pool2d(latents, kernel_size=5, stride=1, padding=2)
        avg = residue.mean(dim=0, keepdim=True)
        return (avg - avg.mean()) / (avg.std() + 1e-5)

    def _bit_sims(self, image_input):
        avg = self._get_residue(image_input)
        _, _, h, w = avg.shape
        sims = []
        for i in range(self.num_bits):
            base = self._base_at(self.freqs[i], self.phases[i], h, w)
            sims.append(
                F.cosine_similarity(avg.reshape(-1), base.reshape(-1), dim=0).item()
            )
        return sims

    def _pack_hw(self):
        sf = self.pipe.vae_scale_factor
        return 2 * (self.height // sf), 2 * (self.width // sf)

    def _make_callback(self, bits, t_lo=200, t_hi=600):
        H, W = self.height, self.width
        vae_sf = self.pipe.vae_scale_factor
        n_ch = self.pipe.transformer.config.in_channels // 4
        pack_h, pack_w = self._pack_hw()
        mod_cache = {}

        def trajectory_callback(pipe, i, t, kwargs):
            latents = kwargs["latents"]
            t_val = t.item() if isinstance(t, torch.Tensor) else float(t)
            if t_lo < t_val < t_hi:
                spatial = pipe._unpack_latents(latents, H, W, vae_sf)
                _, _, h, w = spatial.shape
                key = (h, w)
                if key not in mod_cache:
                    mod_cache[key] = self._modifier_at(h, w, bits)
                spatial = spatial + self.strength * mod_cache[key].to(spatial.dtype)
                latents = pipe._pack_latents(spatial, spatial.shape[0], n_ch, pack_h, pack_w)
            kwargs["latents"] = latents
            return kwargs

        return trajectory_callback

    def _prepare_img2img_latents(self, init_img, denoising_strength, num_inference_steps=28):
        """用 FluxPipeline + strength 切片模拟 FluxImg2Img（当前 diffusers 无该 class）。"""
        pipe = self.pipe
        device, dtype = self.device, self.dtype
        H, W = self.height, self.width

        image = pipe.image_processor.preprocess(init_img, height=H, width=W)
        image = image.to(device=device, dtype=dtype)
        z = pipe.vae.encode(image).latent_dist.mode()
        shift = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        image_latents = (z - shift) * pipe.vae.config.scaling_factor

        pack_h, pack_w = self._pack_hw()
        image_seq_len = (pack_h // 2) * (pack_w // 2)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        # use_dynamic_shifting=True 时必须传 mu（经 retrieve_timesteps **kwargs）
        timesteps, num_inference_steps = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, device, None, sigmas, mu=mu
        )

        init_timestep = min(int(num_inference_steps * denoising_strength), num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = timesteps[t_start * pipe.scheduler.order :]
        if len(timesteps) < 1:
            raise ValueError(f"denoising_strength={denoising_strength} 过小，无有效步数")

        noise = torch.randn(image_latents.shape, device=device, dtype=dtype)
        latents = pipe.scheduler.scale_noise(image_latents, timesteps[:1], noise)
        n_ch = pipe.transformer.config.in_channels // 4
        latents = pipe._pack_latents(latents, latents.shape[0], n_ch, pack_h, pack_w)
        return latents, timesteps

    def _run_img2img(self, prompt, init_img, bits, denoising_strength, guidance_scale=3.5):
        import diffusers.pipelines.flux.pipeline_flux as flux_pipe_mod

        num_steps = 28
        latents, _ = self._prepare_img2img_latents(
            init_img, denoising_strength, num_inference_steps=num_steps
        )
        cb = self._make_callback(bits)
        # 0.30.x scheduler 不支持自定义 timesteps；在 retrieve 后按 strength 切片。
        _orig = flux_pipe_mod.retrieve_timesteps
        strength = float(denoising_strength)

        def _retrieve(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
            ts, n = _orig(scheduler, num_inference_steps, device, None, sigmas, **kwargs)
            init_timestep = min(int(n * strength), n)
            t_start = int(max(n - init_timestep, 0))
            ts = ts[t_start * scheduler.order :]
            if hasattr(scheduler, "set_begin_index"):
                scheduler.set_begin_index(t_start * scheduler.order)
            return ts, len(ts)

        flux_pipe_mod.retrieve_timesteps = _retrieve
        try:
            return self.pipe(
                prompt=prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=num_steps,
                latents=latents,
                guidance_scale=guidance_scale,
                callback_on_step_end=cb,
            ).images[0]
        finally:
            flux_pipe_mod.retrieve_timesteps = _orig

    @torch.no_grad()
    def extract(self, image_input):
        sims = self._bit_sims(image_input)
        thr = self.extract_threshold
        bits = "".join("1" if s > thr else "0" for s in sims)
        return self._bits_to_msg(bits), bits

    @torch.no_grad()
    def detect(self, image_input):
        sims = self._bit_sims(image_input)
        score = float(np.mean(np.abs(sims)))
        return {
            "is_watermarked": score > 0.002,
            "confidence_score": round(score, 6),
        }


def _load_flux_pipe(model_id, device):
    return FluxPipeline.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)


class FluxText2ImgWatermarker(_FluxWatermarkBase):
    def __init__(self, model_id, device="cuda", **kwargs):
        pipe = _load_flux_pipe(model_id, device)
        super().__init__(pipe, device=device, **kwargs)

    @torch.no_grad()
    def embed(self, prompt, message, denoising_strength=0.06):
        del denoising_strength
        bits = self._msg_to_bits(message)
        cb = self._make_callback(bits)
        return self.pipe(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=28,
            guidance_scale=3.5,
            callback_on_step_end=cb,
        ).images[0]


class FluxImg2ImgWatermarker(_FluxWatermarkBase):
    def __init__(self, model_id, strength=0.05, device="cuda", extract_threshold=0.03, **kwargs):
        pipe = _load_flux_pipe(model_id, device)
        super().__init__(
            pipe, strength=strength, device=device, extract_threshold=extract_threshold, **kwargs
        )

    @torch.no_grad()
    def embed(self, input_image, message, denoising_strength=0.8):
        bits = self._msg_to_bits(message)
        init = input_image if isinstance(input_image, Image.Image) else Image.open(input_image)
        init = init.convert("RGB").resize((self.width, self.height))
        return self._run_img2img("", init, bits, denoising_strength, guidance_scale=3.5)


class FluxImgEditWatermarker(_FluxWatermarkBase):
    def __init__(self, model_id, strength=0.003, device="cuda", **kwargs):
        pipe = _load_flux_pipe(model_id, device)
        super().__init__(pipe, strength=strength, device=device, **kwargs)

    @torch.no_grad()
    def embed(self, input_image, message, prompt, denoising_strength=0.8):
        bits = self._msg_to_bits(message)
        init = input_image if isinstance(input_image, Image.Image) else Image.open(input_image)
        init = init.convert("RGB").resize((self.width, self.height))
        return self._run_img2img(prompt, init, bits, denoising_strength, guidance_scale=3.5)

# 兼容旧名称
FluxFlowTrajectoryWatermarker = FluxImg2ImgWatermarker
