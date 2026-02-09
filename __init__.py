
import importlib.util
import os
import sys
import json

# Print version information when loading
print("\033[92mNakuNode V3.0\033[0m \033[93m---\033[0m \033[1;37mNakuNode is build by Naku.\033[0m It can make your work more easier.")

# List of all nodes and their functions
print("NakuNode V3.0 包含以下节点:")
print("  - NakuNode_SaveImage: 保存图像，支持自定义文件名前缀、路径、格式和质量设置")
print("  - NakuNode_常用尺寸: 提供常用的图像尺寸比例和画面模式选择")
print("  - NakuNode_图像边框: 为图像添加指定颜色和宽度的边框")
print("  - NakuNode_图像标注: 智能标注节点，支持在图像上添加数字标注点")
print("  - NakuNode_文件管理: 文件管理节点，支持批量重命名图像文件")
print("  - NakuNode_图像标注节点V2: 图像标注助手节点，支持透明图层合成")
print("  - NakuNode_简易画板: 简易画板节点，支持自由绘制")
print("  - NakuNode_文本选择器: 文本选择器节点，支持从文本选项列表中选择")
print("  - NakuNode_动态文本拆分与选择: 动态文本拆分与选择节点，适用于Lora提示词筛选器")
print("  - NakuNode_故事板输出: 故事板输出节点，将多个图像组合成网格布局")
print("  - NakuNode_图像组合: 图像组合节点，将两张图像按横向或纵向排列组合")
print("  - API节点: 基于Comfly重新编译的API节点，支持多种AI服务")
print("  - Flux2节点: 专为Flux2模型设计的图像参考节点")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "web"
python = sys.executable

def get_ext_dir(subpath=None, mkdir=False):
    dir = os.path.dirname(__file__)
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    dir = os.path.abspath(dir)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def serialize(obj):
    if isinstance(obj, (str, int, float, bool, list, dict, type(None))):
        return obj
    return str(obj)


py = get_ext_dir("py")
files = os.listdir(py)
all_nodes = {}
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]
    try:
        imported_module = importlib.import_module(".py.{}".format(name), __name__)
        if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
        if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
        if imported_module and hasattr(imported_module, 'NODE_CLASS_MAPPINGS') and hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            serialized_CLASS_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_CLASS_MAPPINGS.items()}
            serialized_DISPLAY_NAME_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_DISPLAY_NAME_MAPPINGS.items()}
            all_nodes[file]={"NODE_CLASS_MAPPINGS": serialized_CLASS_MAPPINGS, "NODE_DISPLAY_NAME_MAPPINGS": serialized_DISPLAY_NAME_MAPPINGS}
    except ImportError as e:
        print(f"Error importing module {name}: {e}")
        pass
    except Exception as e:
        print(f"Unexpected error loading module {name}: {e}")
        pass

# Load API nodes
try:
    imported_module = importlib.import_module(".py.Naku_API", __name__)
    if hasattr(imported_module, 'NODE_CLASS_MAPPINGS'):
        NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    if hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **imported_module.NODE_DISPLAY_NAME_MAPPINGS}
    if imported_module and hasattr(imported_module, 'NODE_CLASS_MAPPINGS') and hasattr(imported_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
        serialized_CLASS_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_CLASS_MAPPINGS.items()}
        serialized_DISPLAY_NAME_MAPPINGS = {k: serialize(v) for k, v in imported_module.NODE_DISPLAY_NAME_MAPPINGS.items()}
        all_nodes["Naku_API"]={"NODE_CLASS_MAPPINGS": serialized_CLASS_MAPPINGS, "NODE_DISPLAY_NAME_MAPPINGS": serialized_DISPLAY_NAME_MAPPINGS}
except ImportError as e:
    print(f"Error importing API module: {e}")
    pass
except Exception as e:
    print(f"Unexpected error loading API module: {e}")
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
