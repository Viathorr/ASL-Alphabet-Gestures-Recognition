import torch
from torch import nn
from src.models.landmarks_branch import LandmarksBranch
from src.models.efficient_net_image_branch import EffSignBranch
from src.models.sign_net_v1_branch import SignNetV1Branch
    
    
class ASLAlphabetClassificationModel(nn.Module):
    def __init__(self, num_classes, landmarks_out_dim=64, image_out_dim=64):
        super().__init__()
        self.landmarks_branch = LandmarksBranch(out_dim=landmarks_out_dim)
        self.sign_image_branch = SignNetV1Branch(out_dim=image_out_dim)

        self.classifier = nn.Linear(in_features=landmarks_out_dim + image_out_dim, out_features=num_classes)

    def forward(self, img, landmarks):
        landmark_features = self.landmarks_branch(landmarks)
        img_features = self.sign_image_branch(img)

        x = torch.cat((landmark_features, img_features), dim=1)

        return self.classifier(x)