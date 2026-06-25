import shutil
import random
from pathlib import Path

RAW_DATA_DIR = Path("kaggle_data/PlantVillage")

OUTPUT_DATA_DIR = Path("dataset")

SELECTED_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

if OUTPUT_DATA_DIR.exists():
    shutil.rmtree(OUTPUT_DATA_DIR)

for split in ["train", "valid", "test"]:
    for class_name in SELECTED_CLASSES:
        (OUTPUT_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

for class_name in SELECTED_CLASSES:
    class_path = RAW_DATA_DIR / class_name

    images=[]
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        images.extend(class_path.glob(extension))
    
    random.shuffle(images)

    num_images = len(images)
    train_end = int(num_images * TRAIN_RATIO)
    val_end = int(num_images * (TRAIN_RATIO + VAL_RATIO))
    
    for image_path in images[:train_end]:
        shutil.copy(image_path, OUTPUT_DATA_DIR / "train" / class_name / image_path.name)
    
    for image_path in images[train_end:val_end]:
        shutil.copy(image_path, OUTPUT_DATA_DIR / "valid" / class_name / image_path.name)
    
    for image_path in images[val_end:]:
        shutil.copy(image_path, OUTPUT_DATA_DIR / "test" / class_name / image_path.name)

    print(f"\n{class_name}")
    print(f"Number of Images: {num_images}")
    print(f"Train: {len(images[:train_end])}")
    print(f"Val: {len(images[train_end:val_end])}")
    print(f"Test: {len(images[val_end:])}")

print("\nDataset preparation completed.")