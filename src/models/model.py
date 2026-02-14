import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
    model = models.mobilenet_v2(weights=None)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model
