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
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # File paths and labels
    '''
    folder_path = 'data/Present&None'

    # Dataset and DataLoader
    dataset = FITSDataset(folder_path, transform=preprocess_fits)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model
    model = get_model(dropout_prob=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training
    train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=30)

    # Evaluation
    evaluate_model(model, val_dataloader, device, criterion)

    # Save the model
    torch.save(model.state_dict(), 'airglow_model.pth')
    '''
    # Load and predict
    model = get_model(dropout_prob=0.5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    folder_path = 'data/PokerFlat_2017_02_01'
    predictions = predict_airglow(model, folder_path, device)
    # Print predictions for all images in the folder
    for filename, prediction in predictions.items():
        print(f'{filename}: {prediction}')
