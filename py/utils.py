from .md import *
import os
import json
import datetime
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import torch
import folder_paths
import scipy.ndimage as ndimage

CATEGORY_TYPE = "NakuNodes/Utils"

class NakuNode_SaveImage:
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

class NakuNode_常用尺寸:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "模型选择": (["Qwen", "Flux2", "WAN", "LTX2"], {"default": "Qwen"}),
                "尺寸选择": (["大尺寸", "小尺寸"], {"default": "大尺寸"}),
                "常用尺寸": (["1:1", "3:2", "4:3", "16:9"], {"default": "1:1"}),
                "画面模式": (["横屏", "竖屏"], {"default": "横屏"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_size"
    CATEGORY = "NakuNodes/Utils"

    def get_size(self, 模型选择, 尺寸选择, 常用尺寸, 画面模式):
        # Define aspect ratio mappings for each model and size
        model_sizes = {
            "WAN": {
                "大尺寸": {
                    "16:9": (1280, 704),
                    "1:1": (1024, 1024),
                    "3:2": (960, 640),
                    "4:3": (1024, 768)
                },
                "小尺寸": {
                    "16:9": (960, 544),
                    "1:1": (512, 512),
                    "3:2": (768, 512),
                    "4:3": (768, 576)
                }
            },
            "LTX2": {
                "大尺寸": {
                    "16:9": (1920, 1088),
                    "1:1": (1024, 1024),
                    "3:2": (1248, 832),
                    "4:3": (1152, 864)
                },
                "小尺寸": {
                    "16:9": (1280, 704),
                    "1:1": (768, 768),
                    "3:2": (768, 512),
                    "4:3": (1152, 864)
                }
            },
            "Qwen": {
                "大尺寸": {
                    "16:9": (1664, 928),
                    "1:1": (1328, 1328),
                    "3:2": (1632, 1088),
                    "4:3": (1472, 1104)
                },
                "小尺寸": {
                    "16:9": (1280, 704),
                    "1:1": (1024, 1024),
                    "3:2": (1248, 832),
                    "4:3": (1280, 960)
                }
            },
            "Flux2": {
                "大尺寸": {
                    "16:9": (1920, 1088),
                    "1:1": (1536, 1536),
                    "3:2": (1920, 1280),
                    "4:3": (1792, 1344)
                },
                "小尺寸": {
                    "16:9": (1280, 704),
                    "1:1": (1280, 1280),
                    "3:2": (1152, 768),
                    "4:3": (1280, 960)
                }
            }
        }

        # Get the dimensions based on model, size, and aspect ratio
        width, height = model_sizes[模型选择][尺寸选择][常用尺寸]

        # If portrait mode is selected, swap width and height
        if 画面模式 == "竖屏":
            width, height = height, width

        return (width, height)


class NakuNode_图像边框:
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


class NakuNode_图像标注:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "标注颜色": (["红色", "蓝色", "黄色", "白色", "黑色"], {"default": "红色"}),
            },
            "hidden": {
                "points_data": ("STRING", {"default": "[]"}),
            },
        }

    RETURN_TYPES = ("COORDINATES", "IMAGE")
    RETURN_NAMES = ("points", "image")
    FUNCTION = "process_image"
    CATEGORY = "NakuNodes/Utils"

    def process_image(self, image, 标注颜色="红色", points_data=None):
        if points_data is None:
            points_data = "[]"

        # 1. 加载图片
        image_path = folder_paths.get_annotated_filepath(image)
        try:
            i = Image.open(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return ([], torch.zeros((1, 512, 512, 3)))

        i = ImageOps.exif_transpose(i)
        image_obj = i.convert("RGB")
        w, h = image_obj.size

        # 2. 解析坐标
        try:
            points = json.loads(points_data)
        except:
            points = []

        # 3. 绘制
        if points:
            draw = ImageDraw.Draw(image_obj)
            try:
                # 字体大小 46 (65 * 0.7 = 45.5，约等于46，缩小30%)
                font = ImageFont.truetype("arial.ttf", 46)
            except:
                # 如果无法加载特定字体，尝试其他常见字体
                try:
                    font = ImageFont.truetype("Arial.ttf", 46)
                except:
                    try:
                        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 46)  # macOS
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 46)  # Linux
                        except:
                            # 如果所有字体都失败，使用默认字体并调整大小
                            font = ImageFont.load_default()
                            # 注意：当使用默认字体时，大小无法调整

            # 定义颜色映射
            color_map = {
                "红色": "#FF0000",
                "蓝色": "#0000FF",
                "黄色": "#FFFF00",
                "白色": "#FFFFFF",
                "黑色": "#000000"
            }
            fill_color = color_map.get(标注颜色, "#FF0000")  # 默认红色

            for idx, p in enumerate(points):
                cx = p['x'] * w
                cy = p['y'] * h

                # 半径 30 (比原来的20增大50%)
                r = 30
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill_color, outline="#FFFFFF", width=3)

                # 绘制数字
                label = str(idx + 1)
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
                    tw, th = right - left, bottom - top
                    # 在圆内居中文字
                    text_x = cx - tw / 2
                    text_y = cy - th / 2
                except:
                    # 当使用默认字体时，尺寸可能无法准确计算，使用预估值
                    tw, th = 38, 38  # 54 * 0.7 = 37.8，约等于38
                    text_x = cx - 19  # 27 * 0.7 = 18.9，约等于19
                    text_y = cy - 19  # 27 * 0.7 = 18.9，约等于19

                draw.text((text_x, text_y), label, fill="#FFFFFF", font=font)

        # 4. 输出
        image_np = np.array(image_obj).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (points, image_tensor)

class NakuNode_文件管理:
    """
    A ComfyUI node for managing files in a directory.
    Allows batch renaming of images with custom prefix and starting number.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_folder": ("STRING", {"default": "", "multiline": False, "placeholder": "输入文件夹路径"}),
                "output_folder": ("STRING", {"default": "", "multiline": False, "placeholder": "输出文件夹路径"}),
                "file_prefix": ("STRING", {"default": "image", "multiline": False, "placeholder": "文件名前缀"}),
                "start_number": ("INT", {"default": 1, "min": 1, "max": 999999, "display": "number"}),
                "file_extension": (["auto", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("输出信息", "输出文件夹")
    FUNCTION = "process_files"
    CATEGORY = "NakuNodes/Utils"

    def process_files(self, input_folder, output_folder, file_prefix, start_number, file_extension):
        import os
        import shutil
        from pathlib import Path
        import re

        # Validate input folder
        if not input_folder or not os.path.isdir(input_folder):
            return (f"Error: 输入文件夹 '{input_folder}' 不存在", "")

        # Validate or create output folder
        if not output_folder:
            output_folder = input_folder  # Use input folder as output if not specified
        else:
            os.makedirs(output_folder, exist_ok=True)

        # Get all image files from input folder
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = []

        for file in os.listdir(input_folder):
            file_path = os.path.join(input_folder, file)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in image_extensions:
                    image_files.append(file)

        # Sort files to ensure consistent renaming
        image_files.sort()

        # Copy and rename files
        renamed_files = []
        current_number = start_number

        for file in image_files:
            old_path = os.path.join(input_folder, file)
            original_ext = os.path.splitext(file)[1]

            # Determine the extension to use
            if file_extension == "auto":
                # Use the original file extension
                final_ext = original_ext
            else:
                # Use the specified extension
                final_ext = f".{file_extension}"

            new_filename = f"{file_prefix}_{current_number:04d}{final_ext}"
            new_path = os.path.join(output_folder, new_filename)

            # Copy file to output folder with new name
            shutil.copy2(old_path, new_path)
            renamed_files.append(new_filename)
            current_number += 1

        # Prepare output info
        info = f"已处理 {len(renamed_files)} 个文件从 '{input_folder}' 到 '{output_folder}'. "
        info += f"文件使用前缀 '{file_prefix}' 从编号 {start_number} 开始重命名。"

        return (info, output_folder)


# --------------------------------------------------------------------------------
# 节点: NakuNode_图像标注节点V2
# --------------------------------------------------------------------------------
class NakuNode_图像标注节点V2:
    """
    一个图像标注助手节点，基于输入图像进行标注。
    它接收原始图像，在前端进行编辑，然后输出最终结果。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 这个字段接收前端画布最终的base64数据
                "annotation_data": ("STRING", {"multiline": True, "default": "data:image/png;base64,"}),
            },
            "optional": {
                "图像": ("IMAGE",), # 输入图像
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "process_annotation"
    CATEGORY = CATEGORY_TYPE

    def process_annotation(self, annotation_data, 图像=None):
        # 确定最终输出图像的尺寸
        if 图像 is not None:
            _, img_h, img_w, _ = 图像.shape
            target_width, target_height = img_w, img_h
        else:
            # 如果没有图像输入，设置默认尺寸
            target_width, target_height = 512, 512

        # 解码前端传来的、包含所有编辑的画布数据
        if annotation_data and annotation_data.strip() and annotation_data != "data:image/png;base64,":
            try:
                base64_str = re.sub(r'^data:image/png;base64,', '', annotation_data)
                decoded_data = base64.b64decode(base64_str)
                # 这是带有透明通道的最终编辑层
                final_edit_layer_pil = Image.open(io.BytesIO(decoded_data)).convert("RGBA")

                # 确保尺寸一致
                if final_edit_layer_pil.size != (target_width, target_height):
                    final_edit_layer_pil = final_edit_layer_pil.resize((target_width, target_height), Image.LANCZOS)
            except Exception as e:
                print(f"[NAKU] 解码标注数据时出错: {e}")
                final_edit_layer_pil = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        else:
            # 如果没有编辑数据，则创建一个完全透明的图层
            final_edit_layer_pil = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))

        # 准备背景图
        if 图像 is not None:
            # 将输入的PyTorch张量转换为Pillow图像
            i = 255. * 图像[0].cpu().numpy()
            bg_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert('RGBA')
        else:
            # 如果没有背景图，则创建一个黑色背景
            bg_pil = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 255))

        # 将编辑图层合成到背景图上
        final_pil = Image.alpha_composite(bg_pil, final_edit_layer_pil)

        # 准备输出：最终合成的图像
        output_np = np.array(final_pil.convert("RGB")).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np)[None,]

        return (output_tensor,)


# --------------------------------------------------------------------------------
# 节点: NakuNode_简易画板
# --------------------------------------------------------------------------------
class NakuNode_简易画板:
    """
    一个简易画板节点，允许用户进行自由绘制。
    它有一个mask输入接口（可选），直接在画布上绘制，然后输出绘制结果。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # 这个字段接收前端画布最终的base64数据
                "canvas_data": ("STRING", {"multiline": True, "default": "data:image/png;base64,"}),
                "画板预设": (["1:1", "3:2", "3:4", "16:9"], {"default": "1:1"}),
                "图像模式": (["横屏", "竖屏"], {"default": "横屏"}),
                "背景颜色": (["白色", "黑色", "灰色"], {"default": "白色"}),
            },
            "optional": {
                "遮罩": ("MASK",),  # 添加mask输入接口
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "process_canvas"
    CATEGORY = CATEGORY_TYPE

    def process_canvas(self, 画板预设, 图像模式, 背景颜色, canvas_data, 遮罩=None):
        # 根据预设和图像模式确定尺寸
        size_map = {
            "1:1": (1328, 1328),
            "3:2": (1584, 1056),
            "3:4": (1140, 1472),
            "16:9": (1664, 928)
        }

        width, height = size_map[画板预设]

        # 如果选择了竖屏模式，交换宽高
        if 图像模式 == "竖屏":
            width, height = height, width

        # 根据背景颜色设置背景
        bg_colors = {
            "白色": (255, 255, 255, 255),
            "黑色": (0, 0, 0, 255),
            "灰色": (128, 128, 128, 255)
        }
        bg_color = bg_colors.get(背景颜色, (255, 255, 255, 255))  # 默认白色

        # 解码前端传来的、包含所有编辑的画布数据
        if canvas_data and canvas_data.strip() and canvas_data != "data:image/png;base64,":
            try:
                base64_str = re.sub(r'^data:image/png;base64,', '', canvas_data)
                decoded_data = base64.b64decode(base64_str)
                # 这是带有透明通道的最终编辑层
                canvas_layer_pil = Image.open(io.BytesIO(decoded_data)).convert("RGBA")

                # 确保尺寸一致
                if canvas_layer_pil.size != (width, height):
                    canvas_layer_pil = canvas_layer_pil.resize((width, height), Image.LANCZOS)
            except Exception as e:
                print(f"[NAKU] 解码画板数据时出错: {e}")
                canvas_layer_pil = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        else:
            # 如果没有编辑数据，则创建一个完全透明的图层
            canvas_layer_pil = Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # 创建指定颜色的背景
        bg_pil = Image.new("RGBA", (width, height), bg_color)

        # 将画板图层合成到背景图上
        final_pil = Image.alpha_composite(bg_pil, canvas_layer_pil)

        # 准备输出：最终合成的图像
        output_np = np.array(final_pil.convert("RGB")).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np)[None,]

        return (output_tensor,)


# --------------------------------------------------------------------------------
# 节点: NakuNode_文本选择器
# --------------------------------------------------------------------------------
class NakuNode_文本选择器:
    """
    A node that allows selection from a list of text options
    """

    def __init__(self):
        self.selected_option = ""

    @classmethod
    def INPUT_TYPES(cls):
        # This creates a dynamic dropdown by using a string that gets parsed
        # For true dynamic dropdowns, we would need to implement a custom widget
        return {
            "required": {
                "options_list": ("STRING", {
                    "multiline": True,
                    "default": "Option 1\nOption 2\nOption 3",
                    "forceInput": True
                }),
                "selected_option_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,  # This will be adjusted based on actual options count
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_text", "selected_index")
    FUNCTION = "select_from_list"
    CATEGORY = CATEGORY_TYPE

    def select_from_list(self, options_list, selected_option_idx=0):
        """
        Select an option from the list based on the index
        """
        # Split the options by newlines
        options = [line.strip() for line in options_list.split('\n') if line.strip()]

        # If there are no options, return empty values
        if not options:
            return ("", 0)

        # Validate the index
        if 0 <= selected_option_idx < len(options):
            selected_text = options[selected_option_idx]
        else:
            # If index is out of bounds, default to first option
            selected_text = options[0]
            selected_option_idx = 0

        return (selected_text, selected_option_idx)


# --------------------------------------------------------------------------------
# 节点: NakuNode_动态文本拆分与选择
# --------------------------------------------------------------------------------
class NakuNode_动态文本拆分与选择:
    """
    用于拆分和选择文本的节点，适用于Lora提示词筛选器
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "1【Lora1】/Lora1提示词\n2【Lora2】/Lora2提示词\n3【Lora3】/Lora3提示词",
                    "placeholder": "请输入待拆分的文本，每行一个选项，格式：序号【模型名称】/模型提示词"
                }),
                "序号选择": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,  # Will be adjusted dynamically in practice
                    "step": 1,
                    "display": "number"
                }),
                "仅输出提示词": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启",
                    "label_off": "关闭"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("选中文本", "全部选项", "实际索引")
    FUNCTION = "process_text"
    CATEGORY = CATEGORY_TYPE

    def process_text(self, text_input, 序号选择=1, 仅输出提示词=True):
        """
        拆分文本并根据索引返回选中的选项
        """
        # 按换行符拆分文本，移除空行
        options = [line.strip() for line in text_input.split('\n') if line.strip()]

        # 如果没有选项，返回空值
        if not options:
            return ("", "", 0)

        # 调整索引（已为1索引，减1以获取数组索引）
        actual_index = 序号选择 - 1
        if 0 <= actual_index < len(options):
            selected_text = options[actual_index]
        else:
            # 如果索引超出范围，默认选择第一个选项
            selected_text = options[0]
            actual_index = 0

        # 如果启用了"仅输出提示词"，提取"/"后面的部分
        if 仅输出提示词 and selected_text:
            parts = selected_text.split('/')
            if len(parts) > 1:
                selected_text = parts[-1].strip()  # 获取最后一个 "/" 后的部分

        # 返回选中的文本、全部选项（以换行符连接）和实际索引
        all_options = "\n".join(options)

        return (selected_text, all_options, actual_index)


# --------------------------------------------------------------------------------
# 节点: NakuNode_故事板输出
# --------------------------------------------------------------------------------
class NakuNode_故事板输出:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # 定义所有支持的布局选项，按第一个数字排序
        layouts = ["2x2", "2x3", "3x2", "3x3", "3x4", "4x3", "4x4", "4x5", "5x4"]

        return {
            "required": {
                "输出格式": (["PNG", "JPEG"],),
                "图片间隔颜色": (["black", "white", "gray"],),
                "布局": (layouts,),
                "长边尺寸": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "间隔像素": ("INT", {"default": 10, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "图片1": ("IMAGE",),
                "图片2": ("IMAGE",),
                "图片3": ("IMAGE",),
                "图片4": ("IMAGE",),
                "图片5": ("IMAGE",),
                "图片6": ("IMAGE",),
                "图片7": ("IMAGE",),
                "图片8": ("IMAGE",),
                "图片9": ("IMAGE",),
                "图片10": ("IMAGE",),
                "图片11": ("IMAGE",),
                "图片12": ("IMAGE",),
                "图片13": ("IMAGE",),
                "图片14": ("IMAGE",),
                "图片15": ("IMAGE",),
                "图片16": ("IMAGE",),
                "图片17": ("IMAGE",),
                "图片18": ("IMAGE",),
                "图片19": ("IMAGE",),
                "图片20": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    FUNCTION = "combine_images"
    CATEGORY = "NakuNodes/Utils"

    def combine_images(self, 输出格式, 图片间隔颜色, 布局, 长边尺寸, 间隔像素, **kwargs):
        # 获取所有传入的图像
        images = []
        # 根据布局计算需要的图片数量，最多检查20个输入
        layout_parts = 布局.split('x')
        if len(layout_parts) == 2:
            rows, cols = int(layout_parts[0]), int(layout_parts[1])
            required_images = rows * cols
        else:
            raise ValueError(f"无效的布局格式: {布局}")

        # 获取指定数量的图片（最多检查20个输入）
        for i in range(1, min(required_images + 1, 21)):
            img_key = f"图片{i}"
            if img_key in kwargs and kwargs[img_key] is not None:
                images.append(kwargs[img_key])

        if not images:
            raise ValueError("至少需要一张图片")

        # 检查图像数量是否足够
        if len(images) < required_images:
            raise ValueError(f"Opps，你只有 {len(images)} 张图片，我需要 {required_images} 张图片哦")

        # 确保图像数量不超过布局容量
        if len(images) > required_images:
            images = images[:required_images]

        # 将PyTorch张量转换为PIL图像，并根据长边设置调整尺寸
        pil_images = []
        for img_tensor in images:
            # 将tensor转换为numpy数组
            i = 255. * img_tensor.cpu().numpy()
            # 去除批次维度 (batch, height, width, channels) -> (height, width, channels)
            img_array = np.squeeze(i, axis=0) if i.shape[0] == 1 else i
            img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

            # 如果设置了长边尺寸，则调整图像大小
            if 长边尺寸 > 0:
                original_width, original_height = img.size
                if original_width > original_height:  # 横向图片
                    if original_width != 长边尺寸:
                        ratio = 长边尺寸 / original_width
                        new_width = 长边尺寸
                        new_height = int(original_height * ratio)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                else:  # 纵向图片
                    if original_height != 长边尺寸:
                        ratio = 长边尺寸 / original_height
                        new_height = 长边尺寸
                        new_width = int(original_width * ratio)
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            pil_images.append(img)

        # 计算单个图像的最大尺寸以适应网格
        max_width = max_height = 0
        for img in pil_images:
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)

        # 创建网格图像
        spacing = 间隔像素  # 使用用户设置的间距像素
        grid_width = cols * max_width + (cols - 1) * spacing
        grid_height = rows * max_height + (rows - 1) * spacing

        # 确定背景色
        if 图片间隔颜色 == "black":
            bg_color = (0, 0, 0)
        elif 图片间隔颜色 == "white":
            bg_color = (255, 255, 255)
        else:  # gray
            bg_color = (128, 128, 128)

        grid_img = Image.new('RGB', (grid_width, grid_height), color=bg_color)

        # 将图像粘贴到网格中
        for idx, img in enumerate(pil_images):
            row = idx // cols
            col = idx % cols

            # 居中放置在网格单元格内
            x_offset = col * (max_width + spacing)
            y_offset = row * (max_height + spacing)

            # 如果图像小于最大尺寸，则居中放置
            paste_x = x_offset + (max_width - img.width) // 2
            paste_y = y_offset + (max_height - img.height) // 2

            grid_img.paste(img, (paste_x, paste_y))

        # 添加边框（边框宽度等于间隔像素，颜色与图片间隔颜色相同）
        if spacing > 0:
            # 创建一个新的图像，尺寸更大，用于添加边框
            bordered_width = grid_width + 2 * spacing
            bordered_height = grid_height + 2 * spacing
            bordered_img = Image.new('RGB', (bordered_width, bordered_height), color=bg_color)
            # 将原图粘贴到新图像中心
            bordered_img.paste(grid_img, (spacing, spacing))
            grid_img = bordered_img

        # 保存图像到临时文件
        import tempfile
        temp_dir = tempfile.mkdtemp()
        filename = f"storyboard_output.{输出格式.lower()}"
        filepath = os.path.join(temp_dir, filename)

        if 输出格式 == "JPEG":
            grid_img = grid_img.convert('RGB')  # JPEG不支持透明度
            grid_img.save(filepath, format="JPEG", quality=95)
        else:  # PNG
            grid_img.save(filepath, format="PNG")

        # 读取保存的图像并转换回tensor
        output_image = Image.open(filepath)
        output_np = np.array(output_image).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np)[None,]

        return (output_tensor, filepath)


# --------------------------------------------------------------------------------
# 节点: NakuNode_图像组合
# --------------------------------------------------------------------------------
class NakuNode_图像组合:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout": (["横排", "竖排"], {"default": "横排"}),
                "title_height": ("INT", {"default": 30, "min": 10, "max": 50, "step": 5}),
                "text_option": (["修改对比", "调色对比", "字母选项"], {"default": "修改对比"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_image",)
    FUNCTION = "combine_images"
    CATEGORY = "NakuNodes/Utils"

    def combine_images(self, layout, title_height, text_option, image1=None, image2=None):
        # 如果两个图像都为空，则返回错误或空图像
        if image1 is None and image2 is None:
            # 创建一个空白图像作为返回值
            empty_image = torch.zeros((1, 100, 100, 3), dtype=torch.float32)
            return (empty_image,)

        # 将PyTorch张量转换为PIL图像
        def tensor_to_pil(tensor):
            # tensor shape: [H, W, C] or [N, H, W, C]
            if len(tensor.shape) == 4:
                tensor = tensor.squeeze(0)
            i = 255. * tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return img

        def pil_to_tensor(img):
            # 将PIL图像转换回PyTorch张量
            img_array = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_array)[None,]
            return tensor

        # 获取文字选项对应的标签
        text_pairs = {
            "横排": ("horizontal", {"修改对比": ["修改前", "修改后"], "调色对比": ["调色前", "调色后"], "字母选项": ["OP1", "OP2"]}),
            "竖排": ("vertical", {"修改对比": ["修改前", "修改后"], "调色对比": ["调色前", "调色后"], "字母选项": ["OP1", "OP2"]})
        }

        # 根据选择确定布局方向和标签
        layout_map = {"横排": "horizontal", "竖排": "vertical"}
        layout_direction = layout_map.get(layout, "horizontal")

        # 获取文字标签
        label_map = {
            "修改对比": ["修改前", "修改后"],
            "调色对比": ["调色前", "调色后"],
            "字母选项": ["OP1", "OP2"]
        }
        labels = label_map.get(text_option, ["", ""])

        # 处理图像
        pil_images = []
        if image1 is not None:
            pil_images.append(tensor_to_pil(image1))
        if image2 is not None:
            pil_images.append(tensor_to_pil(image2))

        # 如果只有一个图像，直接添加标题栏并返回
        if len(pil_images) == 1:
            combined_img = self.add_title_bar(pil_images[0], title_height, labels[0])
            return (pil_to_tensor(combined_img),)

        # 合并两个图像
        if layout_direction == "horizontal":
            combined_img = self.combine_horizontal(pil_images[0], pil_images[1], title_height, labels)
        else:
            combined_img = self.combine_vertical(pil_images[0], pil_images[1], title_height, labels)

        return (pil_to_tensor(combined_img),)

    def add_title_bar(self, img, title_height, label):
        """为单个图像添加标题栏"""
        width, height = img.size

        # 创建新图像，包含标题栏
        new_img = Image.new('RGB', (width, height + title_height), color='black')
        new_img.paste(img, (0, title_height))

        # 添加文字
        draw = ImageDraw.Draw(new_img)

        # 尝试使用系统字体，如果不可用则使用默认字体
        try:
            font_size = max(12, int(title_height * 0.6))  # 字体大小根据标题栏高度调整
            font = ImageFont.truetype("Arial Unicode.ttf", font_size)  # macOS
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)  # Linux
            except IOError:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)  # Windows
                except IOError:
                    font = ImageFont.load_default()  # 默认字体

        # 计算文字位置（左对齐，垂直居中）
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]  # 获取文字的实际高度
        text_x = 5  # 左边距5像素
        # 垂直居中：计算文字绘制的y坐标，使文字在标题栏中垂直居中
        text_y = (title_height - text_height) // 2

        # 绘制文字
        draw.text((text_x, text_y), label, fill="white", font=font)

        return new_img

    def combine_horizontal(self, img1, img2, title_height, labels):
        """水平合并两个带标题栏的图像"""
        titled_img1 = self.add_title_bar(img1, title_height, labels[0])
        titled_img2 = self.add_title_bar(img2, title_height, labels[1])

        # 确保两个图像高度相同
        total_width = titled_img1.width + titled_img2.width
        max_height = max(titled_img1.height, titled_img2.height)

        # 创建新的组合图像
        combined_img = Image.new('RGB', (total_width, max_height), color='black')

        # 将图像粘贴到中心位置（如果高度不同）
        combined_img.paste(titled_img1, (0, (max_height - titled_img1.height) // 2))
        combined_img.paste(titled_img2, (titled_img1.width, (max_height - titled_img2.height) // 2))

        return combined_img

    def combine_vertical(self, img1, img2, title_height, labels):
        """垂直合并两个带标题栏的图像"""
        titled_img1 = self.add_title_bar(img1, title_height, labels[0])
        titled_img2 = self.add_title_bar(img2, title_height, labels[1])

        # 确保两个图像宽度相同
        max_width = max(titled_img1.width, titled_img2.width)
        total_height = titled_img1.height + titled_img2.height

        # 创建新的组合图像
        combined_img = Image.new('RGB', (max_width, total_height), color='black')

        # 将图像粘贴到中心位置（如果宽度不同）
        combined_img.paste(titled_img1, ((max_width - titled_img1.width) // 2, 0))
        combined_img.paste(titled_img2, ((max_width - titled_img2.width) // 2, titled_img1.height))

        return combined_img


class NakuNode_MultiText:
    """
    一个多文本节点，具有三个文本框和三个文本输出接口
    当合并文本选项为True时，将三个文本框的内容合并输出到第一个输出接口
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {"multiline": True, "default": "输入文本1\n\n\n\n", "placeholder": "输入文本1", "dynamicPrompts": True}),
                "text2": ("STRING", {"multiline": True, "default": "输入文本2\n\n\n\n", "placeholder": "输入文本2", "dynamicPrompts": True}),
                "text3": ("STRING", {"multiline": True, "default": "输入文本3\n\n\n\n", "placeholder": "输入文本3", "dynamicPrompts": True}),
                "合并文本": ("BOOLEAN", {"default": False, "label_on": "开启", "label_off": "关闭"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text1_output", "text2_output", "text3_output")
    FUNCTION = "process_texts"
    CATEGORY = CATEGORY_TYPE

    def process_texts(self, text1, text2, text3, 合并文本):
        if 合并文本:
            # 如果启用合并文本，则将三个文本框的内容以换行形式合并到第一个输出
            merged_text = "\n".join(filter(None, [text1, text2, text3]))
            return (merged_text, "", "")
        else:
            # 如果不启用合并，则分别输出到对应的输出接口
            return (text1, text2, text3)


class NakuNodeAssetsCombine:
    """
    图片拼接节点
    支持根据模板拼接最多6张图片，具有多种自定义选项
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 拼接模板选择
                "template_type": (["场景图拼接", "角色图拼接"], {"default": "场景图拼接"}),
                # 拼接方向
                "direction": (["横向拼接", "竖向拼接", "2x3网格拼接"], {"default": "横向拼接"}),
                # 长边像素
                "long_side_pixels": ("INT", {"default": 1024, "min": 512, "max": 4096}),
                # 边框宽度
                "border_width": ("INT", {"default": 30, "min": 20, "max": 100}),
                # 边框颜色
                "border_color": (["黑色", "白色", "红色", "黄色", "蓝色"], {"default": "黑色"}),
                # 输出格式
                "output_format": (["png", "JPEG"], {"default": "png"}),
            },
            "optional": {
                # 最多6张图片输入
                "image_front": ("IMAGE",),
                "image_left": ("IMAGE",),
                "image_right": ("IMAGE",),
                "image_high_angle": ("IMAGE",),
                "image_detail_1": ("IMAGE",),
                "image_detail_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("拼接结果",)
    FUNCTION = "combine_images"
    CATEGORY = CATEGORY_TYPE

    def get_system_font(self, font_size):
        """获取系统支持中文的字体"""
        import platform
        system = platform.system()

        # 尝试多种字体，按照优先级顺序
        font_paths = []

        if system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/PingFang.ttc",  # 苹方
                "/System/Library/Fonts/Helvetica.ttc",  # Helvetica
                "/System/Library/Fonts/STHeiti Light.ttc",  # 华文黑体
                "/System/Library/Fonts/STSong.ttc",  # 华文宋体
                "/System/Library/Fonts/Songti.ttc",  # 宋体
                "/System/Library/Fonts/Hiragino Sans GB.ttc",  # 冬青黑体简
                "Arial Unicode.ttf",  # Arial Unicode
            ]
        elif system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑粗体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/msjh.ttc",  # 微软正黑
                "C:/Windows/Fonts/calibri.ttf",  # Calibri
                "C:/Windows/Fonts/arial.ttf",  # Arial
                "C:/Windows/Fonts/Arial.ttf",  # Arial (大写)
            ]
        else:  # Linux
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK
                "/usr/share/fonts/truetype/noto-cjk/NotoSansCJK-Bold.ttc",  # Noto CJK Bold
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",  # Noto CJK Bold
                "/usr/share/fonts/TTF/DejaVuSans.ttf",  # DejaVu Sans (alternative path)
            ]

        # 尝试加载字体
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, font_size)
            except:
                continue

        # 如果都没有找到，返回默认字体
        return ImageFont.load_default()

    def combine_images(self, template_type, direction, long_side_pixels, border_width, border_color, output_format,
                      image_front=None, image_left=None, image_right=None,
                      image_high_angle=None, image_detail_1=None, image_detail_2=None):

        # 将颜色名称转换为RGB值
        color_map = {
            "黑色": (0, 0, 0),
            "白色": (255, 255, 255),
            "红色": (255, 0, 0),
            "黄色": (255, 255, 0),
            "蓝色": (0, 0, 255)
        }

        border_rgb = color_map.get(border_color, (0, 0, 0))

        # 获取所有输入的图片
        input_images = []
        # 使用固定中文标签
        labels = [
            "正面视角 Front View",
            "左侧视角 Left Side View",
            "右侧视角 Right Side View",
            "高角度 High Angle Shot",
            "细节图1 Detail01",
            "细节图2 Detail02"
        ]

        image_inputs = [image_front, image_left, image_right, image_high_angle, image_detail_1, image_detail_2]

        for img_tensor in image_inputs:
            if img_tensor is not None:
                # 将PyTorch张量转换为PIL图像
                i = 255. * img_tensor.cpu().numpy().squeeze()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                input_images.append(img)
            else:
                input_images.append(None)

        # 过滤掉None图片
        valid_images = [(img, label) for img, label in zip(input_images, labels) if img is not None]

        if not valid_images:
            raise ValueError("至少需要一张输入图片")

        # 计算拼接布局
        if direction == "横向拼接":
            combined_img = self._horizontal_layout(valid_images, border_width, border_rgb, long_side_pixels)
        elif direction == "竖向拼接":
            combined_img = self._vertical_layout(valid_images, border_width, border_rgb, long_side_pixels)
        else:  # 2x3网格拼接
            # 即使图片数量不足6张，也要执行网格布局，缺少的图片用空白或复制现有图片填充
            combined_img = self._grid_layout(valid_images, border_width, border_rgb, long_side_pixels)

        # 将PIL图像转换回PyTorch张量
        combined_tensor = torch.from_numpy(np.array(combined_img).astype(np.float32) / 255.0).unsqueeze(0)

        return (combined_tensor,)

    def _horizontal_layout(self, valid_images, border_width, border_rgb, long_side_pixels):
        """横向拼接布局 - 只在图片上方和图片间添加边框"""
        total_images = len(valid_images)

        # 缩放所有图片，使每张图片的长边等于指定像素
        resized_images = []
        for (img, label) in valid_images:
            # 获取原始尺寸
            orig_width, orig_height = img.size
            # 计算缩放比例
            if orig_width > orig_height:
                # 宽大于高，以宽度为准
                scale_factor = long_side_pixels / orig_width
                new_width = long_side_pixels
                new_height = int(orig_height * scale_factor)
            else:
                # 高大于等于宽，以高度为准
                scale_factor = long_side_pixels / orig_height
                new_height = long_side_pixels
                new_width = int(orig_width * scale_factor)

            # 缩放图片
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append((resized_img, label))

        # 计算最终画布尺寸
        total_width = sum(img.size[0] for img, _ in resized_images) + (len(resized_images) - 1) * (border_width // 2)
        total_height = max(img.size[1] for img, _ in resized_images) + border_width

        # 创建画布 - 使用边框颜色作为背景色，避免白色边框
        canvas = Image.new('RGB', (total_width, total_height), border_rgb)

        # 粘贴图片并添加标签
        x_offset = 0
        font_size = max(12, border_width // 2)

        # 加载支持中文的字体
        font = self.get_system_font(font_size)

        # 计算反色
        text_rgb = tuple(255 - c for c in border_rgb)

        for i, (img, label) in enumerate(resized_images):
            # 在图片上方绘制边框区域
            draw = ImageDraw.Draw(canvas)

            # 绘制图片上方的标签区域
            label_area = [x_offset, 0, x_offset + img.size[0], border_width]
            draw.rectangle(label_area, fill=border_rgb)

            # 添加标签文本，居左对齐
            text_x = x_offset + 5  # 左边距5像素
            text_y = (border_width - font_size) // 2

            # 使用更好的文本渲染方法
            try:
                draw.text((text_x, text_y), label, fill=text_rgb, font=font)
            except UnicodeEncodeError:
                # 如果遇到编码错误，尝试使用ASCII字符
                safe_label = label.encode('utf-8', errors='ignore').decode('utf-8')
                draw.text((text_x, text_y), safe_label, fill=text_rgb, font=font)

            # 粘贴图片
            canvas.paste(img, (x_offset, border_width))

            # 如果不是最后一张图片，在图片右侧添加分隔边框
            x_offset += img.size[0]
            if i < len(resized_images) - 1:  # 不是最后一张图片
                # 分隔边框宽度为设定值的50%
                separator_width = border_width // 2
                # 只在图片上方区域添加分隔线
                separator_area = [x_offset, 0, x_offset + separator_width, border_width]
                draw.rectangle(separator_area, fill=border_rgb)

                x_offset += separator_width

        return canvas

    def _vertical_layout(self, valid_images, border_width, border_rgb, long_side_pixels):
        """竖向拼接布局 - 只在图片上方添加边框"""
        total_images = len(valid_images)

        # 缩放所有图片，使每张图片的长边等于指定像素
        resized_images = []
        for (img, label) in valid_images:
            # 获取原始尺寸
            orig_width, orig_height = img.size
            # 计算缩放比例
            if orig_width > orig_height:
                # 宽大于高，以宽度为准
                scale_factor = long_side_pixels / orig_width
                new_width = long_side_pixels
                new_height = int(orig_height * scale_factor)
            else:
                # 高大于等于宽，以高度为准
                scale_factor = long_side_pixels / orig_height
                new_height = long_side_pixels
                new_width = int(orig_width * scale_factor)

            # 缩放图片
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append((resized_img, label))

        # 计算最终画布尺寸
        total_width = max(img.size[0] for img, _ in resized_images)
        total_height = sum(img.size[1] for img, _ in resized_images) + len(resized_images) * border_width + (len(resized_images) - 1) * (border_width // 2)

        # 创建画布 - 使用边框颜色作为背景色，避免白色边框
        canvas = Image.new('RGB', (total_width, total_height), border_rgb)

        # 粘贴图片并添加标签
        y_offset = 0
        font_size = max(12, border_width // 2)

        # 加载支持中文的字体
        font = self.get_system_font(font_size)

        # 计算反色
        text_rgb = tuple(255 - c for c in border_rgb)

        for i, (img, label) in enumerate(resized_images):
            # 在图片上方绘制边框区域
            draw = ImageDraw.Draw(canvas)

            # 绘制图片上方的标签区域
            label_area = [0, y_offset, total_width, y_offset + border_width]
            draw.rectangle(label_area, fill=border_rgb)

            # 添加标签文本，居左对齐
            text_x = 5  # 左边距5像素
            text_y = y_offset + (border_width - font_size) // 2

            # 使用更好的文本渲染方法
            try:
                draw.text((text_x, text_y), label, fill=text_rgb, font=font)
            except UnicodeEncodeError:
                # 如果遇到编码错误，尝试使用ASCII字符
                safe_label = label.encode('utf-8', errors='ignore').decode('utf-8')
                draw.text((text_x, text_y), safe_label, fill=text_rgb, font=font)

            # 粘贴图片
            canvas.paste(img, (0, y_offset + border_width))

            # 更新y偏移量
            y_offset += border_width + img.size[1]

            # 如果不是最后一张图片，在图片下方添加分隔边框
            if i < len(resized_images) - 1:  # 不是最后一张图片
                # 分隔边框宽度为设定值的50%
                separator_height = border_width // 2
                # 只在标签区域添加分隔线
                separator_area = [0, y_offset, total_width, y_offset + separator_height]
                draw.rectangle(separator_area, fill=border_rgb)

                y_offset += separator_height

        return canvas

    def _grid_layout(self, valid_images, border_width, border_rgb, long_side_pixels):
        """2x3网格拼接布局 - 四周统一有边框"""
        # 如果图片数量不足6张，用空白图片填充
        num_images = len(valid_images)
        if num_images < 6:
            # 创建一个空白图片用于填充
            if num_images > 0:
                first_img = valid_images[0][0]  # 使用第一张图片的尺寸作为参考
                blank_img = Image.new('RGB', first_img.size, (200, 200, 200))  # 灰色背景，与实际图片相同尺寸
            else:
                # 如果没有有效图片，创建一个默认大小的空白图片
                blank_img = Image.new('RGB', (long_side_pixels, long_side_pixels), (200, 200, 200))
            blank_label = "空位"

            # 用现有图片或空白图片填充到6张
            filled_valid_images = valid_images[:]
            for i in range(6 - num_images):
                filled_valid_images.append((blank_img, blank_label))
        else:
            filled_valid_images = valid_images[:6]  # 只取前6张

        # 缩放所有图片，使每张图片的长边等于指定像素
        resized_images = []
        for (img, label) in filled_valid_images:
            # 获取原始尺寸
            orig_width, orig_height = img.size
            # 计算缩放比例
            if orig_width > orig_height:
                # 宽大于高，以宽度为准
                scale_factor = long_side_pixels / orig_width
                new_width = long_side_pixels
                new_height = int(orig_height * scale_factor)
            else:
                # 高大于等于宽，以高度为准
                scale_factor = long_side_pixels / orig_height
                new_height = long_side_pixels
                new_width = int(orig_width * scale_factor)

            # 缩放图片
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append((resized_img, label))

        # 计算网格布局尺寸
        # 2列3行 - 计算每列和每行的最大尺寸
        col1_width = max(resized_images[i][0].size[0] for i in [0, 2, 4])  # 第一列的最大宽度
        col2_width = max(resized_images[i][0].size[0] for i in [1, 3, 5])  # 第二列的最大宽度
        row1_height = resized_images[0][0].size[1]  # 第一行的高度
        row2_height = resized_images[2][0].size[1]  # 第二行的高度
        row3_height = resized_images[4][0].size[1]  # 第三行的高度

        # 计算最终画布尺寸
        # 总宽度 = 左边框 + 第一列 + 中间边框 + 第二列 + 右边框
        total_width = border_width + col1_width + border_width + col2_width + border_width
        # 总高度 = 上边框 + 第一行 + 行间边框 + 第二行 + 行间边框 + 第三行 + 下边框
        total_height = border_width + row1_height + border_width + row2_height + border_width + row3_height + border_width

        # 创建画布 - 使用边框颜色
        canvas = Image.new('RGB', (total_width, total_height), border_rgb)

        font_size = max(12, border_width // 2)

        # 加载支持中文的字体
        font = self.get_system_font(font_size)

        # 计算反色
        text_rgb = tuple(255 - c for c in border_rgb)

        # 定义位置：2列3行，确保边框宽度一致
        # 每个单元格的标签区域位于图片上方，高度为border_width
        positions = [
            (border_width, border_width),  # 第1张 - 左上 [0] - (左边框, 上边框)
            (border_width + col1_width + border_width, border_width),  # 第2张 - 右上 [1] - (左边框+第一列+中间边框, 上边框)
            (border_width, border_width + row1_height + border_width), # 第3张 - 左中 [2] - (左边框, 上边框+第一行+行间边框)
            (border_width + col1_width + border_width, border_width + row1_height + border_width), # 第4张 - 右中 [3] - (左边框+第一列+中间边框, 上边框+第一行+行间边框)
            (border_width, border_width + row1_height + border_width + row2_height + border_width), # 第5张 - 左下 [4] - (左边框, 上边框+第一行+行间边框+第二行+行间边框)
            (border_width + col1_width + border_width, border_width + row1_height + border_width + row2_height + border_width) # 第6张 - 右下 [5] - (左边框+第一列+中间边框, 上边框+第一行+行间边框+第二行+行间边框)
        ]

        # 绘制每个单元格
        for i in range(6):
            img, label = resized_images[i]
            pos_x, pos_y = positions[i]

            # 绘制标签区域（整个单元格的顶部边框区域）
            draw = ImageDraw.Draw(canvas)

            # 标签区域的坐标
            cell_width = col1_width if i % 2 == 0 else col2_width
            cell_height = row1_height if i < 2 else (row2_height if i < 4 else row3_height)

            # 标签区域：在图片上方绘制边框色块
            label_area = [pos_x, pos_y, pos_x + cell_width, pos_y + border_width]
            draw.rectangle(label_area, fill=border_rgb)

            # 添加标签文本，居左对齐
            text_x = pos_x + 5  # 左边距5像素
            text_y = pos_y + (border_width - font_size) // 2

            # 使用更好的文本渲染方法
            try:
                draw.text((text_x, text_y), label, fill=text_rgb, font=font)
            except UnicodeEncodeError:
                # 如果遇到编码错误，尝试使用ASCII字符
                safe_label = label.encode('utf-8', errors='ignore').decode('utf-8')
                draw.text((text_x, text_y), safe_label, fill=text_rgb, font=font)

            # 粘贴图片
            canvas.paste(img, (pos_x, pos_y + border_width))

        # 确保底部边框存在 - 在最下方添加一行边框
        # 实际上，由于画布高度已经包含了底部边框，所以底部边框应该自然存在
        # 如果仍有问题，我们可以明确绘制底部边框
        draw = ImageDraw.Draw(canvas)
        bottom_border_area = [0, total_height - border_width, total_width, total_height]
        draw.rectangle(bottom_border_area, fill=border_rgb)

        return canvas


# 合并所有节点映射
NODE_CLASS_MAPPINGS = {
    "NakuNode_SaveImage": NakuNode_SaveImage,
    "NakuNode_常用尺寸": NakuNode_常用尺寸,
    "NakuNode_图像边框": NakuNode_图像边框,
    "NakuNode_图像标注": NakuNode_图像标注,
    "NakuNode_文件管理": NakuNode_文件管理,
    "NakuNode_图像标注节点V2": NakuNode_图像标注节点V2,
    "NakuNode_简易画板": NakuNode_简易画板,
    "NakuNode_文本选择器": NakuNode_文本选择器,
    "NakuNode_动态文本拆分与选择": NakuNode_动态文本拆分与选择,
    "NakuNode_故事板输出": NakuNode_故事板输出,
    "NakuNode_图像组合": NakuNode_图像组合,
    "NakuNode_MultiText": NakuNode_MultiText,
    "NakuNodeAssetsCombine": NakuNodeAssetsCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NakuNode_常用尺寸": "NakuNode_常用尺寸",
    "NakuNode_图像边框": "NakuNode_图像边框",
    "NakuNode_图像标注": "NakuNode_图像标注",
    "NakuNode_文件管理": "NakuNode_文件管理",
    "NakuNode_图像标注节点V2": "NakuNode_图像标注节点V2",
    "NakuNode_简易画板": "NakuNode_简易画板",
    "NakuNode_文本选择器": "NakuNode_文本选择器",
    "NakuNode_动态文本拆分与选择": "NakuNode_动态文本拆分与选择",
    "NakuNode_故事板输出": "NakuNode_故事板输出",
    "NakuNode_图像组合": "NakuNode_图像组合",
    "NakuNode_MultiText": "NakuNode_MultiText",
    "NakuNodeAssetsCombine": "NakuNode_图片拼接",
    "NakuNode_VideoSave": "NakuNode_视频保存",
}