import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Generate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    report = classification_report(all_labels, all_predictions, target_names=["No Airglow", "Airglow"])
    print("Classification Report:\n", report)
