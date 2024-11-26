import torch.nn as nn
from torchvision.models import resnet50

def get_model():
    model = resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model