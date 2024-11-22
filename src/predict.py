import torch
from preprocess import preprocess_fits
from astropy.io import fits

def predict_airglow(model, file_path, device):
    model.eval()

    hdul = fits.open(file_path)
    image_data = hdul[0].data
    hdul.close()

    image_data = preprocess_fits((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)))
    image_data = image_data.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_data)
        prediction = (output > 0.5).item()

    return 'Airglow Present' if prediction == 1 else 'No Airglow'
