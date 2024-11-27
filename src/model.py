import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model():
    # Use the default weights for ResNet50
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Replace the fully connected layer with your custom layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    return model