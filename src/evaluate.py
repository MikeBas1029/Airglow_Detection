import torch
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, dataloader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0  # Initialize a variable to track the loss

    all_labels = []  # To store all labels
    all_predictions = []  # To store all predictions

    with torch.no_grad():  # No need to track gradients during evaluation
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(
                1)  # Ensure labels have the right shape

            # Forward pass
            outputs = model(images)

            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            # Convert outputs to predictions: apply threshold of 0.5 for binary classification
            predictions = (outputs > 0.5).float()

            # Calculate accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            # Collect labels and predictions for confusion matrix and classification report
            all_labels.extend(labels.cpu().numpy())  # move labels to CPU and convert to numpy array
            all_predictions.extend(predictions.cpu().numpy())  # move predictions to CPU and convert to numpy array

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    # Print the average loss if criterion is provided
    if criterion:
        avg_val_loss = val_loss / len(dataloader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    # Classification Report
    report = classification_report(all_labels, all_predictions, target_names=["No Airglow", "Airglow"])
    print("Classification Report:\n", report)