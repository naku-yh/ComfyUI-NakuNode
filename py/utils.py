from .md import *
import os
import json
import datetime
from PIL import Image, ImageDraw
import numpy as np
import torch
import folder_paths
import scipy.ndimage as ndimage

CATEGORY_TYPE = "NakuNodes/Utils"

class SaveImage_NakuNodes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_{timestamp}",
                    "multiline": False,
                    "tooltip": "文件名前缀，支持表达式：{timestamp}时间戳、{date}日期、{time}时间、{datetime}日期时间、{batch}批次号、{counter}计数器"
                }),
                "path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "留空使用默认输出目录",
                    "tooltip": "保存路径，支持绝对路径和相对路径，不存在时自动创建"
                }),
                "format": (["png", "jpg", "webp"], {
                    "default": "png",
                    "tooltip": "图像保存格式：PNG无损、JPG/WebP有损压缩"
                }),
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "图像质量(1-100)，仅对JPG和WebP格式有效，PNG格式忽略此参数"
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE

    def save_images(self, images, filename_prefix="ComfyUI_{timestamp}", path="", format="png", quality=95):
        # 确定保存路径
        if path:
            # 支持相对路径和绝对路径
            if os.path.isabs(path):
                save_dir = path
            else:
                # 相对路径基于ComfyUI根目录
                save_dir = os.path.join(os.getcwd(), path)
        else:
            # 使用默认输出目录
            save_dir = folder_paths.get_output_directory()

        # 创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 一次性获取时间信息（避免重复计算）
        now = datetime.datetime.now()
        timestamp = str(int(now.timestamp()))
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        datetime_str = now.strftime("%Y%m%d_%H%M%S")

        # 预处理文件名前缀，只替换非批次相关的变量
        base_prefix = filename_prefix.replace("{timestamp}", timestamp)
        base_prefix = base_prefix.replace("{date}", date_str)
        base_prefix = base_prefix.replace("{time}", time_str)
        base_prefix = base_prefix.replace("{datetime}", datetime_str)

        file_extension = f".{format}"

        # 使用类似系统的计数器逻辑
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(base_prefix, save_dir, images[0].shape[1], images[0].shape[0])

        for batch_number, image in enumerate(images):
            # 转换tensor为PIL图像
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            # 处理批次和计数器变量
            processed_prefix = base_prefix
            if "{batch}" in filename_prefix:
                processed_prefix = processed_prefix.replace("{batch}", f"{batch_number:05d}")
            if "{counter}" in filename_prefix:
                processed_prefix = processed_prefix.replace("{counter}", f"{counter:05d}")

            # 生成文件名
            final_filename = f"{processed_prefix}{file_extension}"

            # 如果没有使用批次或计数器变量，且有多张图片，需要避免重名
            if len(images) > 1 and "{batch}" not in filename_prefix and "{counter}" not in filename_prefix:
                name_without_ext = os.path.splitext(final_filename)[0]
                final_filename = f"{name_without_ext}_{batch_number:05d}{file_extension}"

            file_path = os.path.join(full_output_folder, final_filename)
            counter += 1

            # 根据格式保存图像，移除optimize减少处理时间
            if format == "png":
                # 使用与系统相同的compress_level
                img.save(file_path, format='PNG', compress_level=4)

            elif format == "jpg":
                # 确保RGB模式（JPEG不支持透明度）
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(file_path, format='JPEG', quality=quality)

            elif format == "webp":
                img.save(file_path, format='WebP', quality=quality)


        return {}

class QWEN常用尺寸_NakuNodes:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "常用尺寸": (["1:1", "3:2", "4:3", "16:9"],),
                "画面模式": (["横屏", "竖屏"],),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "NakuNodes/Utils"

    def get_size(self, 常用尺寸, 画面模式):
        # Define aspect ratio mappings
        aspect_ratios = {
            "1:1": (1328, 1328),
            "3:2": (1584, 1056),
            "4:3": (1472, 1140),
            "16:9": (1664, 928)
        }

        width, height = aspect_ratios[常用尺寸]

        # If portrait mode is selected, swap width and height
        if 画面模式 == "竖屏":
            width, height = height, width

        return (width, height)


NODE_CLASS_MAPPINGS = {
    "SaveImage_NakuNodes": SaveImage_NakuNodes,
    "QWEN常用尺寸_NakuNodes": QWEN常用尺寸_NakuNodes,
}

class Outline_NakuNodes:
    """
    Outline 节点 (V1.0)
    功能：
    1. 为输入的图像添加指定颜色和宽度的边框
    2. 支持多种预设颜色选择
    3. 可调节边框像素大小
    4. 输出透明背景的图片
    """

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入端口"""
        return {
            "required": {
                "图像": ("IMAGE",),
                "边框颜色": (["白色", "黑色", "红色", "黄色", "蓝色", "绿色"], {"default": "白色"}),
                "边框像素": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("带边框的图像",)

    FUNCTION = "execute"
    CATEGORY = "NakuNodes/Utils"

    def tensor_to_pil(self, tensor_image):
        """转换Tensor为PIL图像"""
        # 确保张量在CPU上
        image_np = tensor_image.cpu().numpy().squeeze(0)
        # 将值域从[0,1]转换到[0,255]
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

        # 检查张量维度来确定图像模式
        if len(tensor_image.shape) == 4:  # Batch, Height, Width, Channels
            if tensor_image.shape[-1] == 4:  # RGBA
                return Image.fromarray(image_np, 'RGBA')
            elif tensor_image.shape[-1] == 3:  # RGB
                return Image.fromarray(image_np, 'RGB')

        # 默认情况，假设是RGB
        return Image.fromarray(image_np, 'RGB')

    def pil_to_tensor(self, pil_image):
        """将PIL图像转回Tensor"""
        # 确保图像是RGBA模式
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        # 转换为numpy数组
        image_np = np.array(pil_image).astype(np.float32) / 255.0

        # 添加batch维度
        return torch.from_numpy(image_np)[None,]

    def get_outline_color(self, color_name):
        """根据颜色名称获取RGB值"""
        color_map = {
            "白色": (255, 255, 255),
            "黑色": (0, 0, 0),
            "红色": (255, 0, 0),
            "黄色": (255, 255, 0),
            "蓝色": (0, 0, 255),
            "绿色": (0, 255, 0)
        }
        return color_map.get(color_name, (255, 255, 255))  # 默认白色

    def execute(self, 图像, 边框颜色, 边框像素):
        # 转换输入图像为PIL格式
        pil_image = self.tensor_to_pil(图像)

        # 确保图像是RGBA模式以处理透明度
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        # 获取边框颜色
        outline_color = self.get_outline_color(边框颜色)

        # 转换为numpy数组进行处理
        np_image = np.array(pil_image)
        height, width = np_image.shape[:2]

        # 创建结果数组，先复制原图
        result_array = np_image.copy()

        # 生成一个表示透明区域的mask
        alpha_channel = np_image[:, :, 3]
        opaque_mask = alpha_channel > 0

        # 逐层扩展描边
        for i in range(边框像素):
            # 使用形态学操作来扩展非透明区域的边界
            # 创建一个3x3的核，用于检测邻近的透明像素
            kernel = np.ones((3, 3), dtype=bool)

            # 扩展不透明区域
            expanded = ndimage.binary_dilation(opaque_mask, structure=kernel)

            # 找到新扩展的区域（在原来透明但现在被扩展到的区域）
            new_outline_positions = expanded & ~opaque_mask

            # 在新扩展的位置添加边框颜色
            result_array[new_outline_positions] = (*outline_color, 255)

            # 更新不透明mask，包含新添加的描边
            opaque_mask = result_array[:, :, 3] > 0

        # 转换回PIL图像
        result_image = Image.fromarray(result_array, 'RGBA')

        # 转换回tensor格式并返回
        return (self.pil_to_tensor(result_image),)


NODE_CLASS_MAPPINGS = {
    "SaveImage_NakuNodes": SaveImage_NakuNodes,
    "QWEN常用尺寸_NakuNodes": QWEN常用尺寸_NakuNodes,
    "Outline_NakuNodes": Outline_NakuNodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImage_NakuNodes": "SaveImage_NakuNodes",
    "QWEN常用尺寸_NakuNodes": "QWEN常用尺寸_NakuNodes",
    "Outline_NakuNodes": "Outline_NakuNodes",
}