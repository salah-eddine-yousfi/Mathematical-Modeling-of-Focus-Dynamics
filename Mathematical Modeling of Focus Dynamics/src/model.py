import torch
import torch.nn as nn
from torchvision import models

def build_model(cfg, device):
    num_classes = len(cfg["classes"])

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features

    dropout_p = float(cfg.get("dropout_p", 0.3))
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_features, num_classes)
    )

    return model.to(device)
