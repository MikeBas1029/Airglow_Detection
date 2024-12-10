import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=30, early_stopping_patience=5):
    model.to(device)

    best_val_loss = float('inf')  # Initialize this variable to a very large value
    epochs_without_improvement = 0

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # Training Loop
        for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = correct_preds / total_preds
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():  # No gradient calculation during validation
            for images, labels in val_dataloader:
                images, labels = images.to(device).float(), labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

            avg_val_loss = val_loss / len(val_dataloader)
            val_accuracy = correct_preds / total_preds
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Check if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the model with the best validation loss
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation loss.")
                break

    print("Training complete.")

    # Plotting
    epochs = range(1, len(train_losses) + 1)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()