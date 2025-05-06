import torch
from torch import nn
from torchvision.models import efficientnet_b0
    
    
class EffSignBranch(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        
        backbone = efficientnet_b0(pretrained=True)
        self.features = backbone.features
        
        for param in backbone.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=backbone.classifier[1].in_features, out_features=out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)

        return self.relu(x)  # (batch_size, 128)-shaped image features