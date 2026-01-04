# ComfyUI-NakuNode

## 鸣谢

本项目中的 FastCanvas 节点基于 GitHub 作者 @LAOGOU-666 (https://github.com/LAOGOU-666) 开源的 fastcanvas 节点进行开发，在此表示感谢！

## 介绍

NakuNode 是一个为 ComfyUI 设计的工具集，提供了一系列实用的图像处理和操作节点，使我们的操作更加直观和便捷。

## 功能特性

### 1. 绘画/标注节点

#### NAKU图像标注节点V2.0 (NAKUAnnotationHelperV2_NakuNodes)
- 基于输入图像进行标注的图像标注助手节点
- 接收原始图像，在前端进行编辑，然后输出最终结果
- 支持透明图层合成

#### NAKU简易画板 (NAKUSimpleCanvas_NakuNodes)
- 允许用户进行自由绘制的简易画板节点
- 支持多种画板预设尺寸和背景颜色选择
- 可选遮罩输入接口

### 2. 智能标注节点

#### NAKU 智能标注 (NAKUSmartAnnotation_NakuNodes)
- 图像标记和标注工具
- 支持多种颜色选择（红色、蓝色、黄色、白色、黑色）
- 标记点半径比原来增大50%
- 标记数字比原来增大170%并加粗显示
- 支持交互式标记（左键添加，Shift+左键删除）

### 3. 文本处理节点

#### Naku文本选择器节点 (NakuTextSelectorNode)
- 允许从文本选项列表中基于索引进行选择
- 支持多行文本输入

#### Naku动态文本拆分与选择节点 (NakuDynamicTextSplitNode)
- 用于拆分和选择文本的节点
- 适用于Lora提示词筛选器
- 支持格式：序号【模型名称】/模型提示词

### 4. 画布节点

#### FastCanvasTool_NakuNodes
- 快速画布工具节点
- 支持多图层操作

#### FastCanvas_NakuNodes
- 快速画布节点（输出节点）
- 提供交互式画布界面

#### FastCanvasComposite_NakuNodes
- 快速画布合成节点
- 支持变换数据和混合模式

#### TransformDataFromString_NakuNodes
- 从字符串创建变换数据
- 支持JSON格式输入

### 5. 工具节点

#### QWEN常用尺寸_NakuNodes
- 提供常用图像尺寸的节点
- 支持多种宽高比（1:1, 3:2, 4:3, 16:9）
- 可选择横屏或竖屏模式

#### Outline_NakuNodes
- 为输入图像添加指定颜色和宽度的边框
- 支持多种预设颜色选择（白色、黑色、红色、黄色、蓝色、绿色）
- 可调节边框像素大小
- 输出透明背景的图片

#### NAKU文件管理系统
- 快速统一修改文件名
- 方便使用NAKU高效打标系统进行LORA训练的前置打标

### 6. API节点
- 基于Comfly重新编译的API节点
- 除了使用Comfly的api地址外，使用“IP”即可填写任意API节点


## 安装

1. 将此项目克隆或下载到您的 ComfyUI 的 `custom_nodes` 目录中
2. 重启 ComfyUI

## 使用方法

所有节点都可以在 ComfyUI 的节点菜单中找到，位于 "NakuNodes" 类别下。

## 版本信息

**NakuNode V2.0.2** ---  NakuNode is build by Naku. It can make your work more easier.
