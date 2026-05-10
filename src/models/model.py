import timm
import torch.nn as nn


class XRayClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)