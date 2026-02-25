
import importlib.util
import os
import sys
import json

# Print version information when loading
print("\033[92mNakuNode V4.5\033[0m \033[93m---\033[0m \033[1;37mNakuNode is build by Naku.\033[0m It can make your work more easier.")

# List of all nodes and their functions
print("NakuNode V4.5 包含以下节点:")
print("  - NakuNode_SaveImage: 保存图像，支持自定义文件名前缀、路径、格式和质量设置")
print("  - NakuNode_常用尺寸: 提供常用的图像尺寸比例和画面模式选择")
print("  - NakuNode_图像边框: 为图像添加指定颜色和宽度的边框")
print("  - NakuNode_图像标注: 智能标注节点，支持在图像上添加数字标注点")
print("  - NakuNode_文件管理: 文件管理节点，支持批量重命名图像文件")
print("  - NakuNode_图像标注节点V2: 图像标注助手节点，支持透明图层合成")
print("  - NakuNode_简易画板: 简易画板节点，支持自由绘制")
print("  - NakuNode_文本选择器: 文本选择器节点，支持从文本选项列表中选择")
print("  - NakuNode_动态文本拆分与选择: 动态文本拆分与选择节点，适用于Lora提示词筛选器")
print("  - NakuNode_MultiText: 多文本节点，具有三个文本框和三个输出接口，支持合并文本功能")
print("  - NakuNode_图片拼接: 支持根据模板拼接最多9张图片，具有多种自定义选项，新增3x3网格模式")
print("  - NakuNode_故事板输出: 故事板输出节点，将多个图像组合成网格布局")
print("  - NakuNode_图像组合: 图像组合节点，将两张图像按横向或纵向排列组合")
print("  - NakuNode_VideoSave: 视频保存节点，将图像序列保存为视频文件，支持多种格式(h264/h265/ProRes422/ProRes422LT/GIF/WebM)")
print("  - NakuNode_ImageSplit: 图像分割节点，可将单张图像按指定行列数切割成多个子图像")
print("  - NakuNode_镜头控制文字版：VNCCS 位置控制节点，通过滑块控制相机位置生成多视角提示词")
print("  - NakuNode_镜头可视化控制：VNCCS 可视化相机控制节点，带可视化 Widget 的交互式相机控制")
print("  - API节点: 基于Comfly重新编译的API节点，支持多种AI services")
print("  - Flux2节点: 专为Flux2模型设计的图像参考节点")
print("  - Flux2AIO 节点：Flux2 一体化节点，集成模型加载、LoRA、KSampler 和 VAE 解码")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "web"

# 国际化支持 - 加载翻译文件
def load_translation(lang="zh"):
    """加载指定语言的翻译文件"""
    locales_dir = os.path.join(os.path.dirname(__file__), "locales", lang)
    if not os.path.exists(locales_dir):
        return {}
    
    translation = {}
    for file in os.listdir(locales_dir):
        if file.endswith(".json"):
            try:
                with open(os.path.join(locales_dir, file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    translation.update(data)
            except Exception as e:
                print(f"Error loading translation {lang}/{file}: {e}")
    return translation

# 加载中英文翻译
TRANSLATIONS = {
    "zh": load_translation("zh"),
    "en": load_translation("en"),
}

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

# 首先导入 md 模块，因为其他模块可能依赖它
md_path = os.path.join(py, "md.py")
if os.path.exists(md_path):
    try:
        spec = importlib.util.spec_from_file_location("py.md", md_path)
        md_module = importlib.util.module_from_spec(spec)
        # 添加到 sys.modules 以便其他模块可以导入它
        sys.modules["py.md"] = md_module
        spec.loader.exec_module(md_module)
    except Exception as e:
        print(f"Error importing md module: {e}")
        import traceback
        traceback.print_exc()

files = os.listdir(py)
all_nodes = {}
for file in files:
    if not file.endswith(".py"):
        continue
    name = os.path.splitext(file)[0]
    # 跳过 md 模块，因为它已经被单独导入
    if name == "md":
        continue
    try:
        # 构建模块规范并加载
        spec = importlib.util.spec_from_file_location(f"py.{name}", os.path.join(py, file))
        imported_module = importlib.util.module_from_spec(spec)
        # 添加到 sys.modules 以便模块内部的相对导入可以工作
        sys.modules[f"py.{name}"] = imported_module
        spec.loader.exec_module(imported_module)
        
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
        import traceback
        traceback.print_exc()
        pass


# Load API nodes
try:
    # 构建模块规范并加载
    api_py = os.path.join(py, "Naku_API.py")
    if os.path.exists(api_py):
        spec = importlib.util.spec_from_file_location("py.Naku_API", api_py)
        imported_module = importlib.util.module_from_spec(spec)
        # 添加到 sys.modules 以便模块内部的相对导入可以工作
        sys.modules["py.Naku_API"] = imported_module
        spec.loader.exec_module(imported_module)
        
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
    import traceback
    traceback.print_exc()
    pass


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ComfyUI 国际化支持 - 导出翻译字典
# ComfyUI 会查找 NODE_DISPLAY_NAME_MAPPINGS 字典来显示节点名称和描述
# 通过在节点目录中添加 locales 文件夹，ComfyUI 会自动加载翻译
