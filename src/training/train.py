import os
import torch
import mlflow
import mlflow.pytorch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/processed"
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 1


def get_data_loaders():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader


def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return val_loss, accuracy, all_labels, all_preds


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    path = "artifacts/confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path


def train():
    mlflow.set_experiment("cats_dogs_experiment")

    with mlflow.start_run():

        model = get_model().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        train_loader, val_loader = get_data_loaders()

        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LR)
        mlflow.log_param("epochs", EPOCHS)

        train_losses = []
        val_losses = []

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

            train_loss = running_loss / len(train_loader)
            val_loss, val_acc, labels, preds = evaluate(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Acc: {val_acc:.4f}")

        # Save model
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/model.pt"
        torch.save(model.state_dict(), model_path)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(model_path)

        # Confusion matrix
        cm_path = plot_confusion_matrix(labels, preds)
        mlflow.log_artifact(cm_path)

        print("Training completed with experiment tracking.")


if __name__ == "__main__":
    train()
