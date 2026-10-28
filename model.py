
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ZeroLabelModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        network = resnet18(weights=weights)

        self.backbone = nn.Sequential(
            *list(network.children())[:-1]
        )

        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, image):
        features = self.backbone(image).flatten(1)
        projections = self.projector(features)
        return F.normalize(projections, dim=1)

    def extract_features(self, image):
        features = self.backbone(image).flatten(1)
        return F.normalize(features, dim=1)
