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

        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(image_data) != np.min(image_data):
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        else:
            image_data = np.zeros_like(image_data)

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, label