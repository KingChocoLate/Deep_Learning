from datasets_ready import Multilabel_Dataset, dataloader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, multilabel_confusion_matrix
import numpy as np
from prepare_data import load_dataframes
from pathlib import Path
import model

EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 6
THRESHOLD = 0.35
NUM_WORKERS=8

DATASET_DIR = Path("dataset")
TRAIN_PATH = DATASET_DIR / "train_split.csv"
VAL_PATH = DATASET_DIR / "validation_split.csv"
TEST_PATH = DATASET_DIR / "test_split.csv"
IMAGE_DIR = DATASET_DIR / "train_images"

CLASS_NAMES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

def transform() -> tuple[transforms, transforms, transforms]:
    train_transform = transforms.Compose([
        transforms.Resize(
            400,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        transforms.RandomCrop(380),

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=15,
            interpolation=InterpolationMode.BICUBIC
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(380, 380)),
        transforms.ToTensor()
    ])

    test_transform = val_transform

    return train_transform, val_transform, test_transform

def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model: torch.nn.Module,
                dataloader: DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                threshold=0.5,
                device=None):
    model.train()
    train_loss, macro_f1_score = 0, 0
    all_labels = []
    all_preds = []

    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        probs = torch.sigmoid(y_logits.detach())
        preds = (probs >= threshold).int()

        all_labels.append(y.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    train_loss /= len(dataloader)

    macro_precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    micro_precision = precision_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0
        )

    micro_recall = recall_score(
        all_labels,
        all_preds,
        average="micro",
            zero_division=0
    )

    macro_f1_score = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )
    micro_f1_score = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    return {
        "train_loss": train_loss,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1_score,
        "macro_f1": macro_f1_score
    }
    
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device,
    threshold=0.5
):
    model.eval()
    all_labels = []
    all_preds = []
    val_loss = 0

    with torch.inference_mode():
        for image, labels in tqdm(dataloader, desc="Validating"):
            image, labels = image.to(device), labels.to(device)

            y_logits = model(image)
            
            loss = loss_fn(y_logits, labels)
            val_loss += loss.item()

            probs = torch.sigmoid(y_logits.detach())
            preds = (probs >= threshold).int()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)

        val_loss /= len(dataloader)

        macro_precision = precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )

        micro_precision = precision_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0
        )

        micro_recall = recall_score(
            all_labels,
            all_preds,
            average="micro",
            zero_division=0
        )

        macro_f1_score = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        micro_f1_score = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    return {
        "val_loss": val_loss,
        "micro_precision": micro_precision,
        "macro_precision": macro_precision,
        "micro_recall": micro_recall,
        "macro_recall": macro_recall,
        "micro_f1": micro_f1_score,
        "macro_f1": macro_f1_score,
        "labels": all_labels,
        "predictions": all_preds
    }

def epochs_training(model,
                    train_dataloader,
                    val_dataloader,
                    loss_fn,
                    optimizer,
                    threshold=0.5,
                    device=None):
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch} / {EPOCHS}")

        train_results = train_model(model=model,
                    dataloader=train_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    threshold=threshold,
                    device=device
                    )

        val_results = evaluate_model(model=model,
                    dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    threshold=threshold,
                    device=device
                    )

        print(f"Threshold = {threshold}")
        print("______________Training Results______________")
        print(f"Training Loss: {train_results['train_loss']}")
        print(f"Training Precision (micro): {train_results['micro_precision']}")
        print(f"Training Precision (macro): {train_results['macro_precision']}")
        print(f"Training Recall (micro): {train_results['micro_recall']}")
        print(f"Training Recall (macro): {train_results['macro_recall']}")
        print(f"Training F1_Score (micro): {train_results['micro_f1']}")
        print(f"Training F1_Score (macro): {train_results['macro_f1']}")

        print("______________Validation Results______________")
        print(f"Validation Loss: {val_results['val_loss']}")
        print(f"Validation Precision (micro): {val_results['micro_precision']}")
        print(f"Validation Precision (macro): {val_results['macro_precision']}")
        print(f"Validation Recall (micro): {val_results['micro_recall']}")
        print(f"Validation Recall (macro): {val_results['macro_recall']}")
        print(f"Validation F1_Score (micro): {val_results['micro_f1']}")
        print(f"Validation F1_Score (macro): {val_results['macro_f1']}")
    
    checkpoint(model=model,
               file_name="multilabel_model_pos_weights_added.pth",
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=EPOCHS,
               val_loss=val_results["val_loss"],
               val_micro_f1=val_results["micro_f1"],
               val_macro_f1=val_results["macro_f1"],
               class_names=CLASS_NAMES,
               save_dir=Path("saved_models"),
               threshold=THRESHOLD
               )

def test_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device,
    class_names,
    threshold=0.5
):
    testing_results = evaluate_model(model, dataloader, loss_fn, device, threshold)

    print("______________Final Test Results______________")
    print(f"Test Loss: {testing_results["val_loss"]}")
    print(f"Test Precision: {testing_results["macro_precision"]}")
    print(f"Test Recall: {testing_results["macro_recall"]}")
    print(f"Test F1_Score: {testing_results["macro_f1"]}")

    print("\nPer-label classification report:\n")
    classi_report = classification_report(
        testing_results["labels"],
        testing_results["predictions"],
        target_names=class_names,
        zero_division=0
    )
    print(classi_report)

    confusion_matrices = multilabel_confusion_matrix(
        testing_results["labels"],
        testing_results["predictions"]
    )

    for class_name, matrix in zip(
        class_names, 
        confusion_matrices
    ):

        tn, fp, fn, tp = matrix.ravel()

        print(f"\n{class_name}")
        print(f"TN: {tn} | FP: {fp}")
        print(f"FN: {fn} | TP: {tp}")

    testing_results["confusion_matrices"] = confusion_matrices

    return testing_results

def checkpoint(
    model,
    file_name,
    loss_fn,
    optimizer, 
    epochs,
    val_loss,
    val_micro_f1,
    val_macro_f1,
    class_names,
    save_dir: Path,
    threshold=0.5
):
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / file_name

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "epochs": epochs,
        "val_loss": val_loss,
        "val_micro_f1": val_micro_f1,
        "val_macro_f1": val_macro_f1,
        "class_names": class_names,
        "num_classes": len(class_names),
        "threshold": threshold
    }

    torch.save(checkpoint, model_path)

def label_distribution(train_df) -> torch.Tensor:
    targets = np.stack(train_df["targets"].values)

    positive = targets.sum(axis=0)
    negative = len(train_df) - positive
    pos_weights = negative / positive
    for i, label in enumerate(CLASS_NAMES):
        print(f"{label} weight: {pos_weights[i]}")

    return torch.from_numpy(pos_weights)

def main():
    device = set_device()
    train_df = load_dataframes(TRAIN_PATH)
    val_df = load_dataframes(VAL_PATH)
    test_df = load_dataframes(TEST_PATH)

    train_transform, val_transform, test_transform = transform()

    train_dataset = Multilabel_Dataset(
        dataframe=train_df,
        image_dir=IMAGE_DIR,
        transform=train_transform
    )
    val_dataset = Multilabel_Dataset(
        dataframe=val_df,
        image_dir=IMAGE_DIR,
        transform=val_transform
    )
    test_dataset = Multilabel_Dataset(
        dataframe=test_df,
        image_dir=IMAGE_DIR,
        transform=test_transform
    )

    train_dataloader, val_dataloader, test_dataloader = dataloader(train_dataset,
                                                                   val_dataset,
                                                                   test_dataset,
                                                                   batch_size=BATCH_SIZE,
                                                                   num_workers=NUM_WORKERS,
                                                                   )

    efficientnet_model = model.Multilabel_Model(NUM_CLASSES).to(device)

    pos_weights = label_distribution(train_df).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    optimizer = torch.optim.Adam(params=efficientnet_model.parameters(),
                            lr=0.001)
    
    epochs_training(model=efficientnet_model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    threshold=THRESHOLD,
                    device=device
                    )

    test_results = test_model(model=efficientnet_model,
               dataloader=test_dataloader,
               loss_fn=loss_fn,
               device=device,
               class_names=CLASS_NAMES,
               threshold=THRESHOLD)

if __name__ == "__main__":
    main()


    

