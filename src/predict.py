import torch
import numpy as np
from .preprocess import preprocess_fits
from astropy.io import fits

def predict_airglow(model, file_path, device):
    model.eval()

    hdul = fits.open(file_path)
    image_data = hdul[0].data
    hdul.close()

    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.max(image_data) != np.min(image_data):
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    else:
        image_data = np.zeros_like(image_data)

    image_data = preprocess_fits(image_data).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_data)
        prediction = (output > 0.5).item()

    return 'Airglow Present' if prediction == 1 else 'No Airglow'