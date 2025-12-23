from .md import *
import ast

CATEGORY_TYPE = "NakuNodes/Utils"

class NakuTextSplitNode:
    """
    用于拆分多行文本到可选选项的ComfyUI节点
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
            },
            "optional": {
                "仅输出提示词": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启",
                    "label_off": "关闭"
                })
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("选中文本", "选项列表", "选中索引")
    FUNCTION = "split_text"
    OUTPUT_NODE = False  # Not an output node
    CATEGORY = CATEGORY_TYPE

    def split_text(self, text_input, 仅输出提示词=True):
        """
        按换行符拆分输入文本并返回选项
        """
        # 按换行符拆分文本，移除空行
        options = [line.strip() for line in text_input.split('\n') if line.strip()]

        # 如果没有选项，返回空值
        if not options:
            return ("", "", 0)

        # 默认返回第一个选项
        selected_text = options[0]

        # 如果启用了"仅输出提示词"，提取"/"后面的部分
        if 仅输出提示词 and selected_text:
            parts = selected_text.split('/')
            if len(parts) > 1:
                selected_text = parts[-1].strip()  # 获取最后一个 "/" 后的部分

        selected_index = 0

        # 返回第一个选项、完整选项列表（以换行符连接）和索引
        options_list_str = "\n".join(options)

        return (selected_text, options_list_str, selected_index)


class NakuTextSelectionNode:
    """
    一个允许从选项列表中进行选择的辅助节点
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 对于此节点，由于无法根据输入动态更新下拉列表，
        # 我们将使用一个文本字段，用户需要在此粘贴选项列表
        return {
            "required": {
                "options_list": ("STRING", {
                    "multiline": True,
                    "default": "1【Lora1】/Lora1提示词\n2【Lora2】/Lora2提示词\n3【Lora3】/Lora3提示词",
                    "placeholder": "在此粘贴您的选项列表，每行一个，格式：序号【模型名称】/模型提示词"
                }),
                "selected_option": ("STRING", {
                    "default": "",
                    "placeholder": "选中的选项将显示在此处"
                }),
            },
            "optional": {
                "仅输出提示词": ("BOOLEAN", {
                    "default": True,
                    "label_on": "开启",
                    "label_off": "关闭"
                })
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("选中文本", "选中索引")
    FUNCTION = "select_option"
    CATEGORY = CATEGORY_TYPE

    def select_option(self, options_list, selected_option, 仅输出提示词=True):
        """
        从提供的列表中选择一个选项
        """
        # 按换行符拆分选项
        options = [line.strip() for line in options_list.split('\n') if line.strip()]

        # 如果没有选项，返回空值
        if not options:
            return ("", 0)

        # 如果提供了 selected_option 并且在选项列表中存在，则使用它
        if selected_option and selected_option in options:
            selected_text = selected_option
            selected_index = options.index(selected_option)
        else:
            # 否则，默认选择第一个选项
            selected_text = options[0]
            selected_index = 0

        # 如果启用了"仅输出提示词"，提取"/"后面的部分
        if 仅输出提示词 and selected_text:
            parts = selected_text.split('/')
            if len(parts) > 1:
                selected_text = parts[-1].strip()  # 获取最后一个 "/" 后的部分

        return (selected_text, selected_index)


class NakuAdvancedTextSplitNode:
    """
    A ComfyUI node that splits multi-line text into a dropdown of selectable options
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "Option 1\nOption 2\nOption 3",
                    "placeholder": "Enter text to split, one option per line"
                }),
            },
            "optional": {
                "refresh_options": ("BOOLEAN", {"default": False, "label_on": "Refresh", "label_off": "Don't Refresh"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("selected_text", "options_list", "selected_index")
    FUNCTION = "split_and_select"
    CATEGORY = CATEGORY_TYPE

    def split_and_select(self, text_input, refresh_options=False):
        """
        Split the input text by newlines and create a dropdown of options
        """
        # Split the text by newlines, removing empty lines
        options = [line.strip() for line in text_input.split('\n') if line.strip()]

        # If there are no options, return empty values
        if not options:
            return ("", "", 0)

        # By default, return the first option
        selected_text = options[0]
        selected_index = 0

        # Return the first option, the full list of options as a string, and the index
        options_list_str = "\n".join(options)

        return (selected_text, options_list_str, selected_index)


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
    "NakuTextSplitNode": NakuTextSplitNode,
    "NakuTextSelectionNode": NakuTextSelectionNode,
    "NakuAdvancedTextSplitNode": NakuAdvancedTextSplitNode,
    "NakuTextSelectorNode": NakuTextSelectorNode,
    "NakuDynamicTextSplitNode": NakuDynamicTextSplitNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NakuTextSplitNode": "Naku文本拆分节点",
    "NakuTextSelectionNode": "Naku文本选择节点",
    "NakuAdvancedTextSplitNode": "Naku高级文本拆分节点",
    "NakuTextSelectorNode": "Naku文本选择器节点",
    "NakuDynamicTextSplitNode": "Naku动态文本拆分与选择节点"
}