from .md import *
import torch
import numpy as np
from PIL import Image
import base64
import io
import re

CATEGORY_TYPE = "NakuNodes/Utils"

# --------------------------------------------------------------------------------
# 节点: NAKU图像标注节点V2.0
# --------------------------------------------------------------------------------
class NAKUAnnotationHelperV2_NakuNodes:
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
# 节点: NAKU简易画板
# --------------------------------------------------------------------------------
class NAKUSimpleCanvas_NakuNodes:
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


NODE_CLASS_MAPPINGS = {
    "NAKUAnnotationHelperV2_NakuNodes": NAKUAnnotationHelperV2_NakuNodes,
    "NAKUSimpleCanvas_NakuNodes": NAKUSimpleCanvas_NakuNodes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NAKUAnnotationHelperV2_NakuNodes": "Naku图像标注节点V2.0",
    "NAKUSimpleCanvas_NakuNodes": "Naku简易画板"
}