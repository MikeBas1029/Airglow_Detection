from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_fits(image_data):
    # Replace NaN and Inf with zeros
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Convert to 8-bit image
    image = Image.fromarray((image_data * 255).clip(0, 255).astype(np.uint8))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)
