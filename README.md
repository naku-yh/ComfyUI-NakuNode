# NakuNodes

这是为ComfyUI设计的工具集，提供了一系列实用的图像处理和操作节点，让我们的操作变得更加直观方便

## 特别鸣谢

本项目中的 FastCanvas 节点基于 GitHub 作者 [@LAOGOU-666](https://github.com/LAOGOU-666) 开源的 fastcanvas 节点进行开发，在此表示感谢！

## 安装说明

1. 确保已安装ComfyUI
2. 将此仓库克隆到ComfyUI的`custom_nodes`目录下：
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/naku-yh/Comfyui_NakuNode
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动ComfyUI
2. 在右键添加节点中，您可以在"NakuNode"类别下找到所有工具节点
3. 将需要的节点拖入工作区并连接使用

## 节点说明

### 图像尺寸调整节点
基于 Qwen 原生分辨率设置的尺寸调整节点

### 自动 Outline 节点 ###
在使用 FastCanvas 节点拼图时，为方便 QwenImageEdit 识别里面的物体，简易使用 Outline 节点自动帮图片增加白边。

### FastCanvas画布节点
> * 1.支持实时调整构图输出图像和选中图层遮罩
>
> * 2.支持批量构图，切换输入图层继承上个图层的位置和缩放
>
> * 3.支持限制画布窗口视图大小的功能，不用担心图片较大占地方了
>
> * 4.图层支持右键辅助功能，变换，设置背景等，点开有惊喜
>
> * 5.支持通过输入端口输入图像，
      支持无输入端口独立使用，
>
> * 6.支持复制，拖拽，以及上传方式对图像处理
>
注意！fastcanvas tool动态输入节点使用方法：
* bg_img输入背景图片，img输入图层图片，可以输入RGB/RGBA图片
* **系统自带的加载图片节点默认输出的是RGB!不是RGBA（带遮罩通道的图片）!使用加载图像输入RGBA需要合并ALPHA图层！**

