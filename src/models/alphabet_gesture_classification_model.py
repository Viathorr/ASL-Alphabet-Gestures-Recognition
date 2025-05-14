import torch
from torch import nn
from src.models.landmarks_branch import LandmarksBranch
from src.models.efficient_net_image_branch import EffSignBranch
from src.models.sign_net_v1_branch import SignNetV1Branch
from src.models.grayscale_image_branch import SignImageBranch
from src.models.inception_sign_branch import SignImageBranch as InceptionSignImageBranch
    

# Grayscale Inception Model
class ASLAlphabetClassificationModel(nn.Module):
    def __init__(self, num_classes, landmarks_out_dim=64, image_out_dim=64):
        super().__init__()
        self.landmarks_branch = LandmarksBranch(out_dim=landmarks_out_dim)
        self.sign_image_branch = InceptionSignImageBranch(in_channels=1, out_dim=image_out_dim)

        # self.classifier = nn.Linear(in_features=landmarks_out_dim + image_out_dim, out_features=num_classes)  # used with sign_net_v1 model
        self.classifier = nn.Sequential(
            nn.Linear(landmarks_out_dim + image_out_dim, 128),            
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )  # used with Grayscale/InceptionSignImageBranch 

    def forward(self, img, landmarks):
        landmark_features = self.landmarks_branch(landmarks)
        img_features = self.sign_image_branch(img)

        x = torch.cat((landmark_features, img_features), dim=1)  # landmark features first for SignNetV1Branch, grayscale InceptionModel

        return self.classifier(x)