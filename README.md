# ComfyUI-NakuNode

**NakuNode V4.5** ---  NakuNode is build by Naku. It can make your work more easier.

## 介绍

NakuNode 是一个为 ComfyUI 设计的影视制作工具集，提供了一系列实用的图像处理和各种操作节点，使我们的操作更加直观和便捷。

<img src="https://github.com/naku-yh/ComfyUI-NakuNode/blob/main/ScreenShot/NakuNode_Tools.png" alt="FLUX2 Image Reference Node Example" width="500"/>

## 功能特性

### 1. 图像处理节点

#### NakuNode_SaveImage
- 保存图像，支持自定义文件名前缀、路径、格式和质量设置

#### NakuNode_图像边框
- 为输入图像添加指定颜色和宽度的边框
- 支持多种预设颜色选择（白色、黑色、红色、黄色、蓝色、绿色）
- 可调节边框像素大小
- 输出透明背景的图片

#### NakuNode_故事板输出
- 将多个图像组合成网格布局，支持自定义间距和输出格式
- 支持多种布局选项（2x2, 2x3, 3x2, 3x3, 3x4, 4x3, 4x4, 4x5, 5x4）
- 可设置长边尺寸和间隔像素

#### NakuNode_图像组合
- 将两张图像按横向或纵向排列组合，并添加标题栏
- 支持多种文字选项（修改对比、调色对比、字母选项）以及自定义文字标题

#### NakuNode_图片拼接
- 支持场景图拼接模式，最多可拼接 9 张图片
- 支持五种拼接模式：横向拼接、竖向拼接、2x2 网格拼接、2x3 网格拼接、3x3 网格拼接
- 可自定义长边像素、边框宽度、边框颜色和输出格式
- 为每张图片自动添加英文标签（Front View / Left Side View / Right Side View / High Angle View / Low Angle View / Back View / Back Side View / Detail01 / Detail02），便于识别不同视角的图像
- 支持自定义标题功能：支持通过前端界面为每张图片设置自定义标题

#### NakuNode_ImageSplit
- 图像分割节点，可将单张图像按指定行列数切割成多个子图像
- 支持多种宽高比选择（16:9, 9:16, 1:1, 4:3, 3:4）
- 可调节收缩像素以避免边缘重叠
- 输出切片图像列表及行列数信息


### 2. 绘画/标注节点

#### NakuNode_图像标注节点V2
- 基于输入图像进行标注的图像标注助手节点
- 接收原始图像，在前端进行编辑，然后输出最终结果
- 支持透明图层合成

#### NakuNode_简易画板
- 允许用户进行自由绘制的简易画板节点
- 支持多种画板预设尺寸和背景颜色选择
- 可选遮罩输入接口

#### NakuNode_图像标注
- 图像标记和标注工具
- 支持多种颜色选择（红色、蓝色、黄色、白色、黑色）
- 支持交互式标记（左键添加，Shift+左键删除）

### 3. 画布节点

#### NakuNode_画布工具
- 支持多图层操作
- 接收背景图像和其他图层图像作为输入

#### NakuNode_画布
- 画布节点（输出节点）
- 提供交互式画布界面
- 支持实时编辑和预览

#### NakuNode_画布合成
- 画布合成节点
- 支持变换数据和混合模式
- 可以对图像进行高级合成操作

#### NakuNode_变换数据
- 从字符串创建变换数据
- 支持JSON格式输入
- 用于定义图像变换参数

### 4. 文本处理节点

#### NakuNode_文本选择器
- 允许从文本选项列表中基于索引进行选择
- 支持多行文本输入

#### NakuNode_动态文本拆分与选择
- 用于拆分和选择文本的节点
- 适用于Lora提示词筛选器
- 支持格式：序号【模型名称】/模型提示词

#### NakuNode_MultiText
- 一个多功能文本节点，具有三个独立的文本框输入
- 提供三个对应的文本输出接口
- 三个文本框默认值分别为："Lora 提示词"、"Prompt 输入"、"Prompt 输入"
- 包含"合并文本"选项，当启用时将三个文本框内容合并到第一个输出接口
- 支持三种合并方式：逗号分割、句号分割、换行合并

### 5. 工具节点

#### NakuNode_常用尺寸
- 提供常用图像尺寸的节点
- 支持多种宽高比（1:1, 3:2, 4:3, 16:9）
- 可选择横屏或竖屏模式
- 支持"自定义尺寸"，使用自定义的长宽数值

#### NakuNode_文件管理
- 文件管理节点，支持批量重命名图像文件
- 快速统一修改文件名
- 方便使用NAKU高效打标系统进行LORA训练的前置打标

### 6. API节点
- 基于Comfly重新编译的API节点
- 支持多种AI服务（Google Veo3, Gemini, Kling, Midjourney, Sora2, Vidu等）
- 除了使用Comfly的api地址外，使用"IP"即可填写任意API节点的URL

### 7. 视频处理节点

#### NakuNode_VideoSave
- 视频保存节点，将图像序列保存为视频文件
- 支持多种视频格式：H.264 MP4、H.265 MP4、ProRes 422 MOV、ProRes 422 LT MOV、WebM、GIF
- 默认帧率设置为25 FPS
- 支持8bit和10bit色深选择
- ProRes格式默认使用10bit色深以获得最佳质量
- 支持音频轨道合并
- 支持元数据嵌入（保留工作流信息和提示词）
- 提供视频预览功能，支持播放、暂停和下载

### 8. Flux2节点

#### NakuNode Flux2AIO
- 专为Flux2模型设计的All in one节点
- 最多支持5张参考图像作为输入
- 支持文生图及图片编辑
- 内置Flux2 Image Reference
- 节点位于 "Flux2" 类别下

#### NakuNode Flux2 Image Reference
- 专为Flux2模型设计的图像参考节点
- 最多支持5张参考图像作为输入
- 使用选定的VAE将图像编码为参考潜空间
- 将参考潜空间的视觉特征与文本条件相结合
- 动态融合视觉和文本特征以增强生成效果
- 可调节强度参数以控制参考图像的影响
- 节点位于 "Flux2" 类别下

### 9. 镜头控制节点

#### NakuNode_镜头控制文字版
- 基于VNCCS 位置控制节点修改，通过滑块控制相机位置生成多视角提示词
- 专为 Qwen-Image-Edit-2511-Multiple-Angles LoRA 优化
- 支持方位角控制（0=正面，90=右侧，180=背面，270=左侧）
- 支持仰角控制（-30=低角度，0=平视，30=高角度，60=俯视）
- 支持拍摄距离选择（特写、中景、广角）
- 可选是否包含<sks>触发词

#### NakuNode_镜头可视化控制
- 基于VNCCS 可视化相机控制节点，带可视化 Widget 的交互式相机控制
- 支持鼠标点击选择相机角度
- 可视化方位角和仰角选择
- 生成用于多视角生成的相机提示词

## 安装

1. 将此项目克隆或下载到您的 ComfyUI 的 `custom_nodes` 目录中
2. 重启 ComfyUI

## 使用方法

所有节点都可以在 ComfyUI 的节点菜单中找到，位于 "NakuNodes" 类别下。Flux2相关节点位于 "Flux2" 类别下。

## 鸣谢

本项目中的 FastCanvas 节点基于 GitHub 作者 @LAOGOU-666 (https://github.com/LAOGOU-666) 开源的 fastcanvas 节点进行开发，在此表示感谢！
本项目中的 VNCCS 节点基于Github 作者@AHEKOT (https://github.com/AHEKOT/ComfyUI_VNCCS) 开源的VNCCS 节点进行开发，在此表示感谢！
