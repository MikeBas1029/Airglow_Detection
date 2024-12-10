from torch.utils.data import Dataset
from astropy.io import fits
import numpy as np
import os


class FITSDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []

        # Define the directories for airglow present and no airglow
        self.airglow_present_dir = os.path.join(folder_path, 'airglow_present')
        self.no_airglow_dir = os.path.join(folder_path, 'no_airglow')

        # Get all .fits files in each directory
        self.airglow_images = [os.path.join(self.airglow_present_dir, fname) for fname in os.listdir(self.airglow_present_dir) if fname.endswith('.FITS')]
        self.no_airglow_images = [os.path.join(self.no_airglow_dir, fname) for fname in os.listdir(self.no_airglow_dir) if fname.endswith('.FITS')]

        # Print the found files for debugging
        print(f"Found {len(self.airglow_images)} airglow present files.")
        print(f"Found {len(self.no_airglow_images)} no airglow files.")

        # Combine images and labels
        self.images = self.airglow_images + self.no_airglow_images
        self.labels = [1] * len(self.airglow_images) + [0] * len(self.no_airglow_images)

        # Ensure we have found some files
        if len(self.images) == 0:
            raise ValueError(f"No .fits files found in either '{self.airglow_present_dir}' or '{self.no_airglow_dir}'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_path = self.images[idx]
        label = self.labels[idx]

        # Read the FITS file
        hdul = fits.open(file_path)
        image_data = hdul[0].data
        hdul.close()

        # Preprocess the image data (e.g., normalization)
        image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(image_data) != np.min(image_data):
            image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
        else:
            image_data = np.zeros_like(image_data)

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, label