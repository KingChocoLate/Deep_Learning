import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

class Multilabel_Dataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 image_dir: Path,
                 transform: transforms=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[idx]
        image_name = row["image"]
        image_path = self.image_dir / image_name

        target = torch.tensor(
            row["targets"],
            dtype=torch.float32
        )

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, target


def dataloader(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            num_workers) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

