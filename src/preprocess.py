from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_fits(image_data):
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the data to [0, 1] if it's not already
    if np.max(image_data) != np.min(image_data):
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    else:
        image_data = np.zeros_like(image_data)  # In case of constant image, fill with zeros

    # Convert to a PIL Image
    image = Image.fromarray((image_data * 255).astype(np.uint8))

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to the required input size for ResNet50
        transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel grayscale image (RGB format)
        transforms.ToTensor(),  # Convert to Tensor and normalize between [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    return transform(image)