# YOLOv8 网络结构与调试流程笔记

## 一、YOLOv8 简介

YOLOv8 是 Ultralytics 发布的最新版本 YOLO 系列目标检测模型，相较于 YOLOv5~YOLOv7，YOLOv8 在结构上引入了更加灵活的模块，并在速度与精度之间取得了较好平衡。

- 开发者：Ultralytics
- 发布时间：2023 年初
- 支持任务：目标检测、实例分割、图像分类、姿态估计

---

## 二、YOLOv8 网络结构

YOLOv8 的整体结构仍采用 **Backbone + Neck + Head** 的形式，但结构更加简洁高效。

### 1. Backbone

用于提取图像的基础特征，YOLOv8 的主干网络包括：

- **Conv (CBS)**：卷积 + BN + SiLU 激活函数
- **C2f 模块**：改进的 CSP 模块（Cross Stage Partial）
- **Downsampling**：通过步长为2的卷积实现空间降采样

> 特点：采用了 C2f 模块代替 YOLOv5 的 C3 模块，提高特征复用和参数效率。

### 2. Neck

用于增强特征表达能力，常见结构包括 FPN、PAN 等。YOLOv8 使用改进的 **FPN-PAN** 架构：

- **Feature Pyramid Network (FPN)**：多尺度特征融合
- **Path Aggregation Network (PAN)**：加强特征向上传递

> 特点：自上而下 + 自下而上的特征融合，提高小目标检测能力。

### 3. Head

- 使用一个统一的 Detect head 进行目标分类、边界框回归、置信度评分。
- 每个特征层都会输出多个 anchor-free 格式的检测框。

> Anchor-free：不再使用固定 anchor box，提高泛化能力。

---

## 三、YOLOv8 与 YOLOv1~YOLOv2 对比

| 特性              | YOLOv1 (2016)        | YOLOv2 (2017)         | YOLOv8 (2023)               |
|-------------------|----------------------|------------------------|-----------------------------|
| Backbone          | GoogLeNet-Inspired   | Darknet-19            | 自定义轻量级（C2f模块）    |
| Anchor 机制       | 无（直接回归）       | 引入 anchor box       | Anchor-free                 |
| 检测层输出格式    | S×S×(B×5 + C)         | 采用 anchor-based 输出 | Anchor-free 特征点预测      |
| 多尺度训练        | ✗                    | ✓                     | ✓                           |
| BatchNorm         | ✗                    | ✓                     | ✓                           |
| 特征融合机制      | ✗                    | ✗                     | FPN + PAN                   |
| 多任务支持        | 仅检测               | 仅检测                | 检测、分割、姿态、分类      |
| 模型结构           | 单一卷积网络         | 改进 CNN + anchor     | 模块化结构 + C2f + Detect   |
| 精度              | 63.4 mAP@0.5         | 76.8 mAP@0.5           | > 50 mAP@0.5:0.95（COCO）    |
| 速度              | 非常快               | 快                    | 更快，支持 ONNX/TensorRT    |
