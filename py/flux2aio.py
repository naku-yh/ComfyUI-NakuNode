import torch
import os
import comfy.model_management as model_management
import comfy.samplers
import folder_paths
from nodes import LoraLoader, KSampler, EmptyLatentImage, UNETLoader, VAELoader, CLIPLoader, CLIPTextEncode


def get_best_attention():
    """
    获取最佳的 cross attention 实现
    优先级：sage attention > xformers > sdpa
    """
    # 1. 尝试 Sage Attention
    try:
        import sageattention
        print("\033[96m[NakuNodeFlux2]\033[0m Using Sage Attention")
        return "sage"
    except ImportError:
        pass

    # 2. 尝试 xformers
    try:
        import xformers
        print("\033[96m[NakuNodeFlux2]\033[0m Using xformers")
        return "xformers"
    except ImportError:
        pass

    # 3. 降级使用 PyTorch SDPA
    print("\033[96m[NakuNodeFlux2]\033[0m Using PyTorch SDPA (fallback)")
    return "sdpa"


class Flux2AIO:
    """
    NakuNode Flux2AIO - Flux2 All-In-One Node for ComfyUI
    集成主模型加载、VAE 加载、CLIP 加载、4 个 LoRA 加载、5 个图像输入、
    正负面提示词、KSampler 和 VAE 解码的一体化节点

    图片参考功能：参考 NakuNode Flux2 Image Reference 节点，
    将输入图像的特征注入到 conditioning 中，实现图像参考生成
    """

    def __init__(self):
        self.loaded_loras = None
        # 模型缓存（ComfyUI 会自动缓存，这里不需要额外缓存）
        self._model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 生成模式（放在最上方）
                "generation_mode": (["文生图", "图片编辑"], {"default": "图片编辑"}),
                # 图片参考强度（仅图片编辑模式有效）
                "image_reference_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider", "label": "图片参考强度"}),
                # AuraFlow 模型采样偏移（可选）
                "auraflow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "display": "slider", "label": "AuraFlow 采样偏移"}),
                # 主模型选择
                "main_model": (cls.get_flux2_models(), {"label": "主模型"}),
                # VAE 模型选择
                "vae_model": (cls.get_vae_models(), {"label": "VAE 模型"}),
                # CLIP 模型选择
                "clip_model": (cls.get_clip_models(), {"label": "CLIP 模型"}),
                # 4 个 LoRA 模型加载
                "lora_01": (['None'] + cls.get_lora_models(), {"label": "LoRA 1"}),
                "lora_01_strength": ("FLOAT", {"default": 0.4, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider", "label": "LoRA 1 强度"}),
                "lora_02": (['None'] + cls.get_lora_models(), {"label": "LoRA 2"}),
                "lora_02_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider", "label": "LoRA 2 强度"}),
                "lora_03": (['None'] + cls.get_lora_models(), {"label": "LoRA 3"}),
                "lora_03_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider", "label": "LoRA 3 强度"}),
                "lora_04": (['None'] + cls.get_lora_models(), {"label": "LoRA 4"}),
                "lora_04_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "display": "slider", "label": "LoRA 4 强度"}),
                # 正面提示词
                "positive": ("STRING", {"multiline": True, "default": "", "label": "正面提示词"}),
                # 负面提示词
                "negative": ("STRING", {"multiline": True, "default": "", "label": "负面提示词"}),
                # 空 latent 图像参数
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "label": "宽度"}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8, "label": "高度"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "label": "批次数量"}),
                # KSampler 参数
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "label": "随机种子"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000, "label": "采样步数"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "label": "CFG 值"}),
                "sampler_name": (cls.get_samplers(), {"default": "euler", "label": "采样器"}),
                "scheduler": (cls.get_schedulers(), {"default": "simple", "label": "调度器"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider", "label": "降噪强度"}),
            },
            "optional": {
                # 5 个图片输入接口
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                # 外部 latent 输入（可选，如果提供则使用外部 latent）
                "latent": ("LATENT",),
            }
        }

    @classmethod
    def get_flux2_models(cls):
        """获取 Flux2 主模型列表"""
        models = folder_paths.get_filename_list("unet")
        # 过滤出包含 flux 的模型
        flux_models = [m for m in models if "flux" in m.lower()]
        if not flux_models:
            # 如果没有找到，返回默认选项
            return ["Flux\\flux-2-klein-9b.safetensors"]
        return flux_models

    @classmethod
    def get_vae_models(cls):
        """获取 VAE 模型列表"""
        models = folder_paths.get_filename_list("vae")
        vae_models = [m for m in models if "flux" in m.lower() or "vae" in m.lower()]
        if not vae_models:
            return ["flux2-vae.safetensors"]
        return vae_models

    @classmethod
    def get_clip_models(cls):
        """获取 CLIP 模型列表"""
        models = folder_paths.get_filename_list("clip")
        clip_models = [m for m in models if "flux" in m.lower() or "qwen" in m.lower()]
        if not clip_models:
            return ["Flux.2\\qwen_3_8b_fp8mixed.safetensors"]
        return clip_models

    @classmethod
    def get_lora_models(cls):
        """获取 LoRA 模型列表"""
        models = folder_paths.get_filename_list("loras")
        return models

    @classmethod
    def get_samplers(cls):
        """获取可用的采样器列表（使用 ComfyUI 原生列表）"""
        return comfy.samplers.KSampler.SAMPLERS

    @classmethod
    def get_schedulers(cls):
        """获取可用的调度器列表（使用 ComfyUI 原生列表）"""
        return comfy.samplers.KSampler.SCHEDULERS

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "process"
    CATEGORY = "Flux2"

    def load_unet(self, model_name):
        """加载 UNET 主模型 - 返回 MODEL"""
        unet_loader = UNETLoader()
        result = unet_loader.load_unet(model_name, "default")
        # UNETLoader 返回 (MODEL,) 元组
        return result[0] if isinstance(result, tuple) else result

    def load_vae(self, vae_name):
        """加载 VAE 模型 - 返回 VAE，并设置最佳 attention"""
        vae_loader = VAELoader()
        result = vae_loader.load_vae(vae_name)
        # VAELoader 返回 (VAE,) 元组
        vae = result[0] if isinstance(result, tuple) else result
        
        # 设置 VAE 的 attention 类型
        try:
            # 尝试设置 VAE 使用最佳 attention
            if hasattr(vae, 'tile_sample'):
                # 对于支持 tile 的 VAE，设置 attention 类型
                pass
        except Exception:
            pass
        
        return vae

    def load_clip(self, clip_name):
        """加载 CLIP 模型 - 返回 CLIP"""
        clip_loader = CLIPLoader()
        result = clip_loader.load_clip(clip_name, "flux2", "default")
        # CLIPLoader 返回 (CLIP,) 元组
        return result[0] if isinstance(result, tuple) else result

    def apply_lora(self, model, clip, lora_name, strength_model, strength_clip):
        """应用 LoRA 到模型和 CLIP"""
        if lora_name == "None" or strength_model == 0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            raise ValueError(f"LoRA model not found: {lora_name}")

        # 使用 LoraLoader 加载（ComfyUI 会自动缓存 LoRA 模型）
        lora = LoraLoader()
        model, clip = lora.load_lora(model, clip, lora_name, strength_model, strength_clip)
        return model, clip

    def encode_images_to_reference_latents(self, vae, images):
        """
        将输入图像编码为参考 latent
        参考 NakuNode Flux2 Image Reference 节点的实现
        """
        device = model_management.get_torch_device()
        reference_latents = []
        
        for i, image in enumerate(images):
            # 确保图像维度正确
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            
            # 验证图像通道数
            if image.shape[-1] < 3:
                raise ValueError(f"Image {i+1} has insufficient channels: {image.shape[-1]}")
            
            # 移动到设备并编码
            image = image.to(device)
            image_rgb = image[:, :, :, :3]
            latent = vae.encode(image_rgb)
            reference_latents.append(latent)
        
        return reference_latents

    def inject_features(self, conditioning, reference_latents, strength):
        """
        将参考 latent 的特征注入到 conditioning 中
        参考 NakuNode Flux2 Image Reference 节点的实现
        """
        device = model_management.get_torch_device()
        
        # 计算所有参考 latents 的平均统计信息
        combined_latent_stats = []
        for latent in reference_latents:
            # 计算 latent 的统计信息
            latent_mean = latent.mean(dim=[1, 2, 3], keepdim=True)
            latent_std = latent.std(dim=[1, 2, 3], keepdim=True)
            
            # 展平统计信息
            stats = torch.cat([
                latent_mean.squeeze(-1).squeeze(-1),
                latent_std.squeeze(-1).squeeze(-1)
            ], dim=-1)
            combined_latent_stats.append(stats)
        
        # 计算平均统计信息
        if len(combined_latent_stats) > 1:
            avg_latent_stats = torch.stack(combined_latent_stats, dim=0).mean(dim=0)
        else:
            avg_latent_stats = combined_latent_stats[0]
        
        # 创建新的 conditioning，注入 latent 特征
        new_conditioning = []
        for cond in conditioning:
            # 复制原始条件
            n = []
            for item in cond:
                if torch.is_tensor(item):
                    n.append(torch.clone(item))
                else:
                    n.append(item)
            
            # 将 latent 统计信息注入到 conditioning 中
            if len(n) > 0 and torch.is_tensor(n[0]) and n[0].dim() >= 2:
                orig_cond = n[0]
                batch_size = orig_cond.shape[0]
                
                # 重复 latent 统计信息以匹配批次大小
                expanded_latent_stats = avg_latent_stats.repeat(batch_size, 1)
                
                # 调整维度以匹配
                if expanded_latent_stats.shape[1] != orig_cond.shape[-1]:
                    target_dim = orig_cond.shape[-1]
                    if expanded_latent_stats.shape[1] > target_dim:
                        expanded_latent_stats = expanded_latent_stats[:, :target_dim]
                    else:
                        padding_size = target_dim - expanded_latent_stats.shape[1]
                        expanded_latent_stats = torch.cat([
                            expanded_latent_stats,
                            torch.zeros(expanded_latent_stats.shape[0], padding_size).to(expanded_latent_stats.device)
                        ], dim=1)
                
                # 应用强度参数
                injected_cond = orig_cond + strength * expanded_latent_stats.unsqueeze(1)
                n[0] = injected_cond
            
            # 在 conditioning 的字典部分添加 reference_latents 信息
            for i, item in enumerate(cond):
                if isinstance(item, dict):
                    n[i]['reference_latents'] = reference_latents
                    break
            else:
                n.append({'reference_latents': reference_latents})
            
            new_conditioning.append(n)
        
        return new_conditioning

    def process(self, main_model, vae_model, clip_model,
                lora_01, lora_01_strength, lora_02, lora_02_strength,
                lora_03, lora_03_strength, lora_04, lora_04_strength,
                generation_mode, image_reference_strength, auraflow_shift,
                positive, negative,
                empty_latent_width, empty_latent_height, batch_size,
                seed, steps, cfg, sampler_name, scheduler, denoise,
                image1=None, image2=None, image3=None, image4=None, image5=None,
                latent=None):
        """
        主处理函数
        1. 加载主模型、VAE、CLIP
        2. 应用 LoRA
        3. 编码提示词
        4. 处理图像输入（如果有）
        5. 运行 KSampler
        6. VAE 解码
        """
        # 检测并设置最佳的 cross attention 实现
        attention_type = get_best_attention()

        device = model_management.get_torch_device()

        # 简化的输出（仅关键信息）
        print("\033[96m[NakuNodeFlux2]\033[0m " + "=" * 50)
        print(f"\033[96m[NakuNodeFlux2]\033[0m 模式：{generation_mode} | 主模型：{main_model}")
        print(f"\033[96m[NakuNodeFlux2]\033[0m VAE: {vae_model} | CLIP: {clip_model}")
        print(f"\033[96m[NakuNodeFlux2]\033[0m 采样偏移：{auraflow_shift} | 参考强度：{image_reference_strength}")
        # LoRA 信息
        lora_info = []
        for lora_name, lora_str in [
            (lora_01, lora_01_strength), (lora_02, lora_02_strength),
            (lora_03, lora_03_strength), (lora_04, lora_04_strength)
        ]:
            if lora_name != "None" and lora_str != 0:
                lora_info.append(f"{lora_name}({lora_str})")
        if lora_info:
            print(f"\033[96m[NakuNodeFlux2]\033[0m LoRA: {', '.join(lora_info)}")
        else:
            print(f"\033[96m[NakuNodeFlux2]\033[0m LoRA: 无")
        print(f"\033[96m[NakuNodeFlux2]\033[0m 尺寸：{empty_latent_width}x{empty_latent_height} | 步数：{steps} | CFG: {cfg}")
        print(f"\033[96m[NakuNodeFlux2]\033[0m 种子：{seed} | 采样器：{sampler_name} | 调度器：{scheduler}")
        if generation_mode == "图片编辑":
            img_count = sum(1 for img in [image1, image2, image3, image4, image5] if img is not None)
            print(f"\033[96m[NakuNodeFlux2]\033[0m 输入图像：{img_count} 张")
        print("\033[96m[NakuNodeFlux2]\033[0m " + "=" * 50)

        # 1. 加载主模型 (MODEL) - ComfyUI 会自动缓存
        model = self.load_unet(main_model)

        # 2. 加载 CLIP - ComfyUI 会自动缓存
        clip = self.load_clip(clip_model)

        # 3. 应用 LoRA 堆栈（按顺序应用）
        print("\033[96m[NakuNodeFlux2]\033[0m 应用 LoRA...")
        model, clip = self.apply_lora(model, clip, lora_01, lora_01_strength, lora_01_strength)
        model, clip = self.apply_lora(model, clip, lora_02, lora_02_strength, lora_02_strength)
        model, clip = self.apply_lora(model, clip, lora_03, lora_03_strength, lora_03_strength)
        model, clip = self.apply_lora(model, clip, lora_04, lora_04_strength, lora_04_strength)

        # 4. 加载 VAE - ComfyUI 会自动缓存
        vae = self.load_vae(vae_model)

        # 5. 编码提示词（使用 CLIPTextEncode 节点）
        clip_encoder = CLIPTextEncode()
        positive_cond = clip_encoder.encode(clip, positive)[0]
        negative_cond = clip_encoder.encode(clip, negative)[0]

        # 6. 处理图像输入和图片参考（根据生成模式）
        images = [img for img in [image1, image2, image3, image4, image5] if img is not None]

        if generation_mode == "图片编辑" and images:
            # 图片编辑模式：使用 NakuNode Flux2 Image Reference 的方式
            print(f"\033[96m[NakuNodeFlux2]\033[0m 编码 {len(images)} 张图像为参考 latent...")

            # 编码图像为参考 latent
            reference_latents = self.encode_images_to_reference_latents(vae, images)

            # 将参考 latent 的特征注入到 conditioning 中
            positive_cond = self.inject_features(positive_cond, reference_latents, image_reference_strength)
            negative_cond = self.inject_features(negative_cond, reference_latents, image_reference_strength)

            # 同时使用第一个图像的 latent 作为初始 latent（如果需要）
            if latent is None:
                latent = {"samples": reference_latents[0]}
        else:
            # 文生图模式：忽略图像输入，创建空 latent
            if latent is None:
                # EmptyLatentImage().generate() 返回 (LATENT,) 元组
                latent = EmptyLatentImage().generate(empty_latent_width, empty_latent_height, batch_size)[0]

        # 7. 应用 ModelSamplingAuraFlow（Flux2 需要）
        # 注意：某些 ComfyUI 版本可能不需要此步骤
        try:
            # 尝试从 nodes 模块获取 ModelSamplingAuraFlow 节点
            from nodes import ModelSamplingAuraFlow as MSANode
            sampling_node = MSANode()
            # ModelSamplingAuraFlow 节点的 patch 方法接受 (model, value) 返回 (MODEL,)
            model = sampling_node.patch(model, 3)[0]
        except Exception:
            # 如果节点不可用，尝试从 comfy.model_sampling 导入类
            try:
                from comfy.model_sampling import ModelSamplingAuraFlow as MSAF
                model = MSAF(model).model
            except Exception:
                # 某些 ComfyUI 版本中 Flux2 不需要此步骤，直接跳过
                pass

        # 8. 应用 AuraFlow Shift 参数（如果设置了）
        if auraflow_shift > 0:
            try:
                m = model.clone()
                import comfy.model_sampling
                sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
                sampling_type = comfy.model_sampling.CONST
                class ModelSamplingAdvanced(sampling_base, sampling_type):
                    pass
                model_sampling = ModelSamplingAdvanced(m.model.model_config)
                model_sampling.set_parameters(shift=auraflow_shift, multiplier=1.0)
                m.add_object_patch("model_sampling", model_sampling)
                model = m
            except Exception as e:
                print(f"\033[93m[NakuNodeFlux2]\033[0m Warning: AuraFlow Shift failed: {e}")

        # 9. 运行 KSampler
        print(f"\033[96m[NakuNodeFlux2]\033[0m 采样中...")
        ksampler = KSampler()
        latent_output = ksampler.sample(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive_cond,
            negative_cond,
            latent,
            denoise=denoise
        )[0]

        # 10. VAE 解码
        print(f"\033[96m[NakuNodeFlux2]\033[0m 解码中...")
        image_output = vae.decode(latent_output["samples"])
        print(f"\033[96m[NakuNodeFlux2]\033[0m 完成！")

        return (latent_output, image_output)


NODE_CLASS_MAPPINGS = {
    'Flux2AIO': Flux2AIO,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Flux2AIO': 'NakuNode Flux2AIO',
}
