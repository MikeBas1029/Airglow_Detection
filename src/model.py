import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_model(dropout_prob=0.5):
    # Use the default weights for ResNet50
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # Remove the final fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),  # Add an intermediate dense layer for better learning
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=dropout_prob),  # Add dropout with the desired probability
        nn.Linear(512, 1),  # Final classification layer
        nn.Sigmoid()  # Sigmoid activation for binary classification
    )

    return model