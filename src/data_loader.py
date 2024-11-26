from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np

class FITSDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        hdul = fits.open(file_path)
        image_data = hdul[0].data
        hdul.close()

        # Avoid divide-by-zero errors
        data_min, data_max = np.min(image_data), np.max(image_data)
        if data_max > data_min:  # Valid range
            image_data = (image_data - data_min) / (data_max - data_min)
        else:
            image_data = np.zeros_like(image_data)  # Default to all zeros if range is invalid

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, label