import os
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/processed"
BATCH_SIZE = 8          # CPU friendly
LR = 0.001
EPOCHS = 1              # Keep 1 for now (we can increase later)


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def train():
    mlflow.set_experiment("cats_dogs_experiment")

    with mlflow.start_run():

        model = get_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_loader, val_loader = get_data_loaders()

        # Log parameters
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)

            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pt"
        torch.save(model.state_dict(), model_path)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(model_path)

        print("Training completed and model saved.")


if __name__ == "__main__":
    train()