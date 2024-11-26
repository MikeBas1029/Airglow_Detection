import torch
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
    # File paths and labels
    file_paths = ['data\PokerFlat_2016_02_17\PKR_DASC_0428_20160217_031732.392.FITS', 'data\PokerFlat_2016_08_11\PKR_DASC_0428_20160811_074552.595.FITS']
    labels = [1, 0]

    # Dataset and DataLoader
    dataset = FITSDataset(file_paths, labels, transform=preprocess_fits)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)

    # Evaluation
    evaluate_model(model, dataloader, device)

    # Save the model
    save_model(model, 'airglow_model.pth')

    # Load and predict
    model = load_model(get_model(), 'airglow_model.pth')
    print(predict_airglow(model, 'data\PokerFlat_2016_08_11\PKR_DASC_0428_20160811_074552.595.FITS', device))
    print(predict_airglow(model, 'data\PokerFlat_2016_02_17\PKR_DASC_0428_20160217_031744.956.FITS', device))