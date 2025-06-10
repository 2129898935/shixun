import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 64x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 64x8x8

            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # 192x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 192x4x4

            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 384x4x4
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 256x4x4
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x4x4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 256x2x2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 测试网络
if __name__ == "__main__":
    net = AlexNet()
    input = torch.randn(64, 3, 32, 32)  # 64张32x32的RGB图像
    output = net(input)
    print(output.shape)  # torch.Size([64, 10])