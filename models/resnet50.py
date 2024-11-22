import torch
from torchvision import models

def get_resnet50(num_classes=2, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model
