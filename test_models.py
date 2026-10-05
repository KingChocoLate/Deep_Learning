import model
import torch
from torch.utils.data import DataLoader
from datasets_ready import Multilabel_Dataset
from pathlib import Path
import training
from prepare_data import load_dataframes

NUM_CLASSES = 6
DATASET_DIR = Path("dataset")
TEST_PATH = DATASET_DIR / "test_split.csv"
IMAGE_DIR = DATASET_DIR / "train_images"
CLASS_NAMES = training.CLASS_NAMES
BATCH_SIZE = 32
NUM_WORKERS = 4

def main():
    device = training.set_device()
    test_transform = training.transform()[2]
    test_df = load_dataframes(TEST_PATH)
    test_dataset = Multilabel_Dataset(dataframe=test_df,
                                      image_dir=IMAGE_DIR,
                                      transform=test_transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)
    
    test_model = model.Multilabel_Model(NUM_CLASSES).to(device)
    
    checkpoint = torch.load("saved_models/multilabel_model_pos_weights_added.pth", weights_only=False)

    test_model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = torch.nn.BCEWithLogitsLoss()

    training.test_model(model=test_model,
                        dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        device=device,
                        class_names=CLASS_NAMES,
                        threshold=0.5
                        )
    
if __name__ == "__main__":
    main()
