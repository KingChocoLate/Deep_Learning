from pathlib import Path
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, confusion_matrix

from utils import get_device, show_image
from model import PlantDiseaseModel

DATA_DIR = Path("dataset")
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"

OUTPUT_DIR = Path("output")
MODEL_DIR = Path("saved_models")

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 2

def create_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor()
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

    return train_transforms, eval_transforms

def load_datasets(train_transforms, eval_transforms):
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(root=VALID_DIR, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transforms)

    return train_dataset, valid_dataset, test_dataset

def create_dataloaders(train_dataset, valid_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=True)

    return train_loader, valid_loader, test_loader

def train_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device):
    model.train()

    train_loss, train_correct = 0, 0

    for images, labels in tqdm(data_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        output_logits = model(images)

        loss = loss_fn(output_logits, labels)
        train_loss += loss.item()
        train_correct += (output_logits.argmax(dim=1) == labels).sum().item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(data_loader)
    train_accuracy = train_correct / len(data_loader.dataset)

    return train_loss, train_accuracy

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device):
    model.eval()

    eval_loss, eval_correct = 0, 0

    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            output_logits = model(images)

            loss = loss_fn(output_logits, labels)
            eval_loss += loss.item()
            eval_correct += (output_logits.argmax(dim=1) == labels).sum().item()

    eval_loss /= len(data_loader)
    eval_accuracy = eval_correct / len(data_loader.dataset)

    return eval_loss, eval_accuracy

def plot_history(history):
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    plt.show()

    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["valid_accuracy"], label="Valid Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png")
    plt.show()

def save_confusion_matrix(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.show()

def test_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    class_names: list
    ):
    test_loss, test_acc = evaluate_model(model, data_loader, loss_fn, device)

    print("\nTest Result")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    all_preds = []
    all_labels = []

    model.eval()

    with torch.inference_mode():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            output_logits = model(images)
            preds = output_logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds, target_names=class_names
    ))

    save_confusion_matrix(all_labels, all_preds, class_names)

def save_model(
    model: torch.nn.Module,
    class_names: list,
    ):

    save_path = MODEL_DIR / "plant_disease_resnet18_model.pth"

    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
    },
    save_path)

    print(f"Model saved to {save_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    train_transforms, eval_transforms = create_transforms()
    train_data, valid_data, test_data = load_datasets(train_transforms, eval_transforms)

    train_loader, valid_loader, test_loader = create_dataloaders(train_data, valid_data, test_data)

    images, labels = next(iter(train_loader))
    show_image(images[0], labels[0], train_data.classes)

    model = PlantDiseaseModel(num_classes=len(train_data.classes), device=device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_model(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        valid_loss, valid_accuracy = evaluate_model(
            model=model,
            data_loader=valid_loader,
            loss_fn=loss_fn,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    plot_history(history)

    test_model(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        class_names=train_data.classes
    )

    save_model(
        model=model,
        class_names=train_data.classes,
    )

if __name__ == "__main__":
    main()

