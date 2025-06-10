# 深度学习基础

完整的深度学习训练套路

![image.png](attachment:42b0a0a2-895c-4b83-85c3-268d79fdf5d5:image.png)

训练一定是两次循环

欠拟合：训练训练数据集表现不好，验证表现不好

过拟合：训练数据训练过程表现得很好，在我得验证过程表现不好

![image.png](attachment:c9f9e8ed-69b4-4830-88fa-023048678d72:image.png)

# 卷积神经网络

卷积过程

```jsx
import torch
import torch.nn.functional as F
input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])
kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 不满足conv2d的尺寸要求
print(input.shape)
print(kernel.shape)

# 尺寸变换
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
print(input.shape)
print(kernel.shape)

output = F.conv2d(input=input,weight=kernel,stride=1)
print(output)

output2 = F.conv2d(input=input,weight=kernel,stride=2)
print(output2)

# padding 在周围扩展一个像素，默认为0；
output3 = F.conv2d(input=input,weight=kernel,stride=1,padding=1)
print(output3)
```

5*5的输入数据 3*3的卷积核 步长1 填充1，

## 

### 图片卷积

```jsx
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset_chen",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset,
                        batch_size=64)

class CHEN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=6,
                               kernel_size=3,
                               stride=1,
                               padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

chen = CHEN()
print(chen)

writer = SummaryWriter("conv_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = chen(imgs)

    # print(imgs.shape)  # torch.Size([64, 3, 32, 32])
    # print(output.shape)  # torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) ->([**, 3, 30, 30])
    output = torch.reshape(output, (-1, 3, 30, 30))  # -1:会根据后面的值进行调整
    writer.add_images("output", output, step)
    step += 1

定义我们的网络模型
```

![image.png](attachment:59df05a7-f114-4c4a-95c5-be72ea153ecc:image.png)

### tensorboard使用

使用之前安装一下tensorboard

这段代码的作用只是为了拿到我的conv_logs里面的文件

使用tensorboard命令打开

tensorboard --logdir=conv_logs

![image.png](attachment:c589f4aa-51e5-4e67-8c5b-cba9a683eb8a:image.png)

点击链接得到一下界面