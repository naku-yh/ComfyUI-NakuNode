from .md import *
import ast

CATEGORY_TYPE = "NakuNodes/Utils"



class NakuTextSelectorNode:
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


class NakuDynamicTextSplitNode:
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


NODE_CLASS_MAPPINGS = {
    "NakuTextSelectorNode": NakuTextSelectorNode,
    "NakuDynamicTextSplitNode": NakuDynamicTextSplitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NakuTextSelectorNode": "Naku文本选择器节点",
    "NakuDynamicTextSplitNode": "Naku动态文本拆分与选择节点"
}