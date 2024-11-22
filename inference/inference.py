import torch
from PIL import Image

def infer(model, image_path, transform, device, classes):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        prediction = torch.argmax(outputs, dim=1)
    return classes[prediction.item()]
