#!/usr/bin/env python
"""
测试新添加的节点是否能正确导入和初始化
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # 测试导入主模块
    import __init__ as naku_nodes
    
    print("✓ 成功导入 NakuNode 主模块")
    
    # 检查新节点是否在映射中
    if "NakuNodeAssetsCombine" in naku_nodes.NODE_CLASS_MAPPINGS:
        print("✓ NakuNodeAssetsCombine 节点已成功注册")
    else:
        print("✗ NakuNodeAssetsCombine 节点未找到")
        
    if "NakuNode_MultiText" in naku_nodes.NODE_CLASS_MAPPINGS:
        print("✓ NakuNode_MultiText 节点已成功注册")
    else:
        print("✗ NakuNode_MultiText 节点未找到")
    
    # 检查显示名称映射
    if "NakuNodeAssetsCombine" in naku_nodes.NODE_DISPLAY_NAME_MAPPINGS:
        print("✓ NakuNodeAssetsCombine 显示名称已注册")
    else:
        print("✗ NakuNodeAssetsCombine 显示名称未找到")
        
    if "NakuNode_MultiText" in naku_nodes.NODE_DISPLAY_NAME_MAPPINGS:
        print("✓ NakuNode_MultiText 显示名称已注册")
    else:
        print("✗ NakuNode_MultiText 显示名称未找到")
    
    # 尝试实例化节点
    AssetsCombineClass = naku_nodes.NODE_CLASS_MAPPINGS["NakuNodeAssetsCombine"]
    MultiTextClass = naku_nodes.NODE_CLASS_MAPPINGS["NakuNode_MultiText"]
    
    assets_combine_instance = AssetsCombineClass()
    multi_text_instance = MultiTextClass()
    
    print("✓ 成功实例化 NakuNodeAssetsCombine")
    print("✓ 成功实例化 NakuNode_MultiText")
    
    # 检查节点的输入类型定义
    assets_input_types = AssetsCombineClass.INPUT_TYPES()
    multi_input_types = MultiTextClass.INPUT_TYPES()
    
    print(f"✓ NakuNodeAssetsCombine INPUT_TYPES: {list(assets_input_types.keys())}")
    print(f"✓ NakuNode_MultiText INPUT_TYPES: {list(multi_input_types.keys())}")
    
    print("\n所有测试通过！新节点已成功集成到项目中。")
    
except ImportError as e:
    print(f"✗ 导入错误: {e}")
except Exception as e:
    print(f"✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()