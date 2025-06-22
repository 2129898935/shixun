# Vision Transformer（ViT）训练自定义图像分类数据集笔记

---

## 1. 训练自定义数据集流程

### 1.1 数据集预处理与划分

使用 `deal_with_datasets.py` 脚本将原始数据集划分为训练集和验证集：

```python
from sklearn.model_selection import train_test_split
import os, shutil

dataset_dir = r'D:\dataset\image2'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
    if class_name not in ['train', 'val']:
        class_path = os.path.join(dataset_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
        train_imgs, val_imgs = train_test_split(images, train_size=0.7, random_state=42)

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.move(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

        shutil.rmtree(class_path)
```
1.2 生成训练/验证路径文件
```python

def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                f.write(f"{img_path} {label}\n")
```
1.3 定义自定义数据集类
```python

from PIL import Image
import torch.utils.data as data

class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, transform):
        self.imgs_path, self.labels = [], []
        with open(txt_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.imgs_path.append(path)
                self.labels.append(int(label))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.imgs_path)
```
1.4 加载数据集
```python

from torchvision import transforms

train_data = ImageTxtDataset('train.txt', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]))
```
2. 使用 GPU 加速训练
2.1 检查 GPU 是否可用
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```
2.2 将模型和数据迁移至 GPU
```python

model = model.to(device)
inputs, labels = inputs.to(device), labels.to(device)
```
2.3 训练主循环示例
```python

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```
2.4 常见问题排查
问题	原因	解决方法
输出维度不匹配	num_classes 设置不正确	修改模型配置中类别数
图像路径读取错误	txt 文件路径不正确	检查是否为绝对路径
GPU 不可用	CUDA 驱动或 PyTorch 版本不匹配	安装正确版本的 PyTorch + CUDA
图像尺寸与 patch 不匹配	patch_size 与 image_size 不整除	调整 image_size 或 patch_size 参数

3. ViT（Vision Transformer）架构与原理
3.1 ViT 简介
ViT（Vision Transformer）由 Google 于 2020 年提出，是一种基于 Transformer 架构的图像分类方法。核心思路是将图像切分为小块（patch），每个 patch 类似于 NLP 中的“词”，通过 Transformer 编码建模图像全局特征。

3.2 ViT 结构详解
3.2.1 Patch Embedding（图像分块）
输入图像（如 224×224）切分为 16×16 大小的 patch

每个 patch 展平后通过线性层映射为向量

公式：

复制
编辑
(224 × 224) / (16 × 16) = 196 个 patch
3.2.2 位置编码（Positional Embedding）
Transformer 无位置信息，需添加位置编码

引入 cls_token 表示整张图像的分类信息

3.2.3 Transformer Encoder
每个 Transformer 块由：

多头自注意力机制（Multi-Head Attention）

前馈网络（Feed Forward MLP）

残差连接 + 层归一化（LayerNorm）

组成，多个块堆叠构建编码器。

3.2.4 MLP Head（分类器）
输出的 cls_token 向量输入到 MLP 层，用于分类预测。

3.3 简化版 ViT 代码（使用 einops）
```python

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim):
        super().__init__()
        patch_dim = patch_size * patch_size * 3
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head=dim//heads, mlp_dim=mlp_dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)  # B x N x D
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n+1]
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])  # 只用 cls_token 结果