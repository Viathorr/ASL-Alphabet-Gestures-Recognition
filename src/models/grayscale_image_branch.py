import torch
from torch import nn


class SignImageBranch(nn.Module):
    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout3 = nn.Dropout(0.25)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=out_dim, kernel_size=3, stride=1, padding="valid"),
            nn.BatchNorm2d(num_features=out_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.dropout3(self.conv_block3(x))
        x = self.conv_block4(x)
        
        x = self.pool(x).squeeze(-1).squeeze(-1)

        return x    