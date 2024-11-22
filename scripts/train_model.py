import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.resnet50 import get_resnet50
from datasets.dataset import get_data_loaders
from training.train import train
import torch.nn as nn
import torch.optim as optim



# Configurations
train_dir = "data/train"
val_dir = "data/val"
batch_size = 32
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loaders
train_loader, val_loader = get_data_loaders(train_dir, val_dir, batch_size)

# Model, Loss, Optimizer
model = get_resnet50(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train
train(model, train_loader, val_loader, criterion, optimizer, device, epochs)
