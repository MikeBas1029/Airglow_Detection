import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Training Loop
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation Loop (Optional, can improve monitoring)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():  # No gradient calculation during validation
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(dataloader)
        print(f"Validation Loss: {avg_val_loss:.4f}")


    print("Training complete.")