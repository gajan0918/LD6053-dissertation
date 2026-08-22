import torch.nn as nn
from torchvision import models


def build_model(num_classes, pretrained=True, fine_tune_blocks=6):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    if fine_tune_blocks > 0:
        for block in model.features[-fine_tune_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
