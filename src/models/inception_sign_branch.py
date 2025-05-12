import torch
from torch import nn


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding="same")
        self.conv3x3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding="same")
        self.conv5x5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding="same")
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, padding="same")
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.pool(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.relu(self.bn(x))

class SignImageBranch(nn.Module):
    def __init__(self, in_channels=3, out_dim=64):
        super().__init__()
        self.conv_block1 = MultiScaleBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv_block2 = MultiScaleBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, out_dim, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.conv_block1(x)))
        x = self.dropout2(self.pool2(self.conv_block2(x)))
        x = self.conv_block3(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x