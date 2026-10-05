import pandas as pd
from pathlib import Path
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import zipfile
from tqdm.auto import tqdm

DATASET_DIR = Path("dataset")
CSV_PATH = DATASET_DIR / "train.csv"

LABEL_NAMES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

def encode_labels(labels: str) -> list[float]:
    image_labels = set(labels.split(" "))
    return [1.0 if label in image_labels else 0.0 for label in LABEL_NAMES]

def load_dataframes(csv_path: Path) -> pd.DataFrame:
    if any(Path("dataset").iterdir()):
        dataframe = pd.read_csv(csv_path)
    else:
        with zipfile.ZipFile('plant-pathology-2021-fgvc8.zip', 'r') as zip_ref:
            zip_ref.extractall('dataset')
        dataframe = pd.read_csv(csv_path)
    dataframe["targets"] = dataframe["labels"].apply(encode_labels)
    return dataframe
    

def create_split(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = np.stack(dataframe["targets"].values)

    # First splitter
    first_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_indices, tem_indices = next(
        first_splitter.split(dataframe["image"].to_numpy(), targets)
    )

    train_df = dataframe.iloc[train_indices]
    tem_df = dataframe.iloc[tem_indices]

    tem_targets = np.stack(tem_df["targets"].values)

    # Second splitter
    second_splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=42
    )

    val_indices, test_indices = next(
        second_splitter.split(
            tem_df["image"].to_numpy(),
            tem_targets
        )
    )

    val_df = tem_df.iloc[val_indices]
    test_df = tem_df.iloc[test_indices]

    return (train_df.reset_index(drop=True), 
            val_df.reset_index(drop=True), 
            test_df.reset_index(drop=True))

def main():
    df = load_dataframes(CSV_PATH)
    train_df, val_df, test_df = tqdm(create_split(df), desc="Splitting")

    train_df.drop(columns=["targets"]).to_csv(
        DATASET_DIR / "train_split.csv",
        index=False
    )
    val_df.drop(columns=["targets"]).to_csv(
        DATASET_DIR / "validation_split.csv",
        index=False
    )
    test_df.drop(columns=["targets"]).to_csv(
        DATASET_DIR / "test_split.csv",
        index=False
    )

if __name__ == '__main__':
    main()



