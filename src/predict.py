import torch
import numpy as np
from .preprocess import preprocess_fits
from astropy.io import fits
import os

def predict_airglow(model, folder_path, device):
    model.eval()
    results = {}

    # Iterate through all FITS files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".FITS"):  # Process only FITS files
            file_path = os.path.join(folder_path, file_name)

            try:
                # Read the FITS file
                hdul = fits.open(file_path)
                image_data = hdul[0].data
                hdul.close()

                # Preprocess the image data
                image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
                if np.max(image_data) != np.min(image_data):
                    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
                else:
                    image_data = np.zeros_like(image_data)

                image_data = preprocess_fits(image_data).unsqueeze(0).to(device)

                # Make the prediction
                with torch.no_grad():
                    output = model(image_data)
                    prediction = (output > 0.5).item()

                # Store the result
                results[file_name] = 'Airglow Present' if prediction == 1 else 'No Airglow'

            except (OSError, ValueError, TypeError) as e:
                print(f"Warning: Skipping file {file_name} due to error: {e}")
                continue

        # Return the results dictionary
    return results