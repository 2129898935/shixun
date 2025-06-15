import os
import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from day03.dataset import ImageTxtDataset

# 开启 CUDA Launch Blocking，方便定位 device-side assert 错误
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert seq_len % patch_size == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x, ps = pack([cls_tokens, x], 'b * d')
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        cls_tokens, _ = unpack(x, ps, 'b * d')
        return self.mlp_head(cls_tokens)

# ------------------------- 数据加载 -------------------------

# 将图像 resize 为 (3, 1, 256)，再在训练循环中展平为 (3, 256)
transform = transforms.Compose([
    transforms.Resize((1, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_data = ImageTxtDataset(
    r"D:\PycharmProjects\day1\day03\data_set\train.txt",
    r"D:\PycharmProjects\day1\day03\data_set\Images\train",
    transform
)
test_data = ImageTxtDataset(
    r"D:\PycharmProjects\day1\day03\data_set\val.txt",
    r"D:\PycharmProjects\day1\day03\data_set\Images\val",
    transform
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64)
print(f"训练集长度: {len(train_data)}, 测试集长度: {len(test_data)}")

# ------------------------- 检查标签范围 & 设置 num_classes -------------------------

# 快速获取标签列表
all_labels = train_data.labels
max_label = max(all_labels)
min_label = min(all_labels)
print(f"标签范围: {min_label} ~ {max_label}")

num_classes = max_label + 1
print(f"自动设置 num_classes = {num_classes}")

# ------------------------- 模型与训练配置 -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(
    seq_len=256,
    patch_size=16,
    num_classes=num_classes,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter("logs_vit")
epoch = 10
total_train_step = 0
total_test_step  = 0
start_time = time.time()

# ------------------------- 训练循环 -------------------------

for i in range(epoch):
    print(f"\n----- 第{i+1}轮训练开始 -----")
    model.train()
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # (B, C, 1, 256) -> (B, C, 256)
        imgs = imgs.squeeze(2)

        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"Step {total_train_step}, train loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 评估
    model.eval()
    total_test_loss = 0.0
    total_correct   = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            imgs = imgs.squeeze(2)

            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            total_correct += (outputs.argmax(1) == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = total_correct / len(test_data)
    print(f"测试 loss: {avg_test_loss:.4f}, 准确率: {test_accuracy:.4f}")

    writer.add_scalar("test_loss", avg_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", test_accuracy, total_test_step)
    total_test_step += 1

    torch.save(model.state_dict(), f"model_save/vit_epoch{i+1}.pth")
    print(f"模型已保存: vit_epoch{i+1}.pth")

writer.close()
print("训练完成！")