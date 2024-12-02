import torch
import os
from src.data_loader import FITSDataset
from src.preprocess import preprocess_fits
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_airglow
from src.utils import save_model, load_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # Directory containing FITS files
    folder_path = 'data/PokerFlat_2018_01_20'

    # Get all FITS files in the directory
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".FITS")]

    # Generate dummy labels (e.g., all zeros since labels are unknown during prediction)
    labels = [0] * len(file_paths)

    # Dataset and DataLoader
    dataset = FITSDataset(file_paths, labels, transform=preprocess_fits)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)

    # Evaluation
    evaluate_model(model, dataloader, device)

    # Save the model
    save_model(model, 'airglow_model.pth')

    # Load and predict
    model = load_model(get_model(), 'airglow_model.pth')
    predictions = predict_airglow(model, folder_path, device)

    # Print predictions for all images in the folder
    for filename, prediction in predictions.items():
        print(f'{filename}: {prediction}')
