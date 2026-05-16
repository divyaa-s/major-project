import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Down
        self.conv1 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)

        # Time projection layers (IMPORTANT)
        self.time_proj1 = nn.Linear(time_dim, base_channels)
        self.time_proj2 = nn.Linear(time_dim, base_channels * 2)

        # Up
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(base_channels, img_channels, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Down 1
        x1 = self.conv1(x)
        t1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)
        x1 = x1 + t1
        x1 = F.relu(x1)

        # Down 2
        x2 = self.pool(x1)
        x2 = self.conv2(x2)
        t2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        x2 = x2 + t2
        x2 = F.relu(x2)

        # Up
        x3 = F.interpolate(x2, scale_factor=2, mode="nearest")
        x3 = self.conv3(x3)
        x3 = F.relu(x3)

        out = self.conv4(x3)
        return out
