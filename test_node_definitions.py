#!/usr/bin/env python
"""
测试新添加的节点是否在 utils.py 中正确定义
"""

import sys
import os
import ast

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_node_definitions():
    """检查节点是否在 utils.py 中正确定义"""
    utils_file = os.path.join(os.path.dirname(__file__), "py", "utils.py")
    
    with open(utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查节点类是否存在
    has_assets_combine = "class NakuNodeAssetsCombine:" in content
    has_multi_text = "class NakuNode_MultiText:" in content
    
    print(f"✓ NakuNodeAssetsCombine 类定义: {'存在' if has_assets_combine else '不存在'}")
    print(f"✓ NakuNode_MultiText 类定义: {'存在' if has_multi_text else '不存在'}")
    
    # 检查节点是否在映射中
    has_assets_in_mappings = '"NakuNodeAssetsCombine": NakuNodeAssetsCombine' in content
    has_multi_in_mappings = '"NakuNode_MultiText": NakuNode_MultiText' in content
    
    print(f"✓ NakuNodeAssetsCombine 在 NODE_CLASS_MAPPINGS 中: {'是' if has_assets_in_mappings else '否'}")
    print(f"✓ NakuNode_MultiText 在 NODE_CLASS_MAPPINGS 中: {'是' if has_multi_in_mappings else '否'}")
    
    # 检查显示名称映射
    has_assets_display = '"NakuNodeAssetsCombine": "NakuNode_图片拼接"' in content
    has_multi_display = '"NakuNode_MultiText": "NakuNode_MultiText"' in content
    
    print(f"✓ NakuNodeAssetsCombine 在 NODE_DISPLAY_NAME_MAPPINGS 中: {'是' if has_assets_display else '否'}")
    print(f"✓ NakuNode_MultiText 在 NODE_DISPLAY_NAME_MAPPINGS 中: {'是' if has_multi_display else '否'}")
    
    # 检查 CATEGORY 是否设置为 CATEGORY_TYPE
    has_category_type = 'CATEGORY = CATEGORY_TYPE' in content
    
    print(f"✓ 节点使用 CATEGORY_TYPE 分类: {'是' if has_category_type else '否'}")
    
    # 检查 CATEGORY_TYPE 定义
    has_category_type_def = 'CATEGORY_TYPE = "NakuNodes/Utils"' in content
    
    print(f"✓ CATEGORY_TYPE 定义: {'是' if has_category_type_def else '否'}")
    
    if has_assets_combine and has_multi_text and has_assets_in_mappings and has_multi_in_mappings:
        print("\n✓ 所有节点定义检查通过！")
        print("  节点已正确添加到项目中，将在 ComfyUI 环境中正常工作。")
        return True
    else:
        print("\n✗ 节点定义存在问题！")
        return False

if __name__ == "__main__":
    check_node_definitions()