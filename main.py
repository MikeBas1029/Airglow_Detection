import torch
from src.data_loader import FITSDataset
from src.preprocess import preprocess_fits
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_airglow
from src.utils import save_model, load_model
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os

if __name__ == "__main__":
    # Paths for Airglow and No Airglow data
    airglow_path = 'data/airglow_present'
    no_airglow_path = 'data/no_airglow'

     # Check if directories exist
    if not os.path.exists(airglow_path):
        raise FileNotFoundError(f"The directory {airglow_path} does not exist.")
    if not os.path.exists(no_airglow_path):
        raise FileNotFoundError(f"The directory {no_airglow_path} does not exist.")

    # Collect FITS file paths and labels
    airglow_files = [os.path.join(airglow_path, f) for f in os.listdir(airglow_path) if f.endswith(".FITS")]
    no_airglow_files = [os.path.join(no_airglow_path, f) for f in os.listdir(no_airglow_path) if f.endswith(".FITS")]

    file_paths = airglow_files + no_airglow_files
    labels = [1] * len(airglow_files) + [0] * len(no_airglow_files)

    # Dataset
    dataset = FITSDataset(file_paths, labels, transform=preprocess_fits)

    # Split dataset into training, validation, and testing sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # Model
    model = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    # Evaluation
    print("Evaluating on Test Data...")
    evaluate_model(model, test_loader, device)

    # Save the model
    save_model(model, 'airglow_model.pth')

    # Load and predict
    model = load_model(get_model(), 'airglow_model.pth')
    predictions = predict_airglow(model, 'data/airglow_present', device)

    # Print predictions
    for filename, prediction in predictions.items():
        print(f'{filename}: {prediction}')
