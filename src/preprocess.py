from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_fits(image_data):
    image = Image.fromarray((image_data * 255).astype(np.uint8))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)