from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models

from PIL import Image
import matplotlib.pyplot as plt

from utils import get_device
from model import PlantDiseaseModel

MODEL_PATH = Path("saved_models/plantdisess.pth")

def load_model(device):
    checkpoint = torch.load(f=MODEL_PATH)
    
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    return model, class_names

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def predict_image(image_path, model, class_names, transform, device):
    image_path = Path(image_path)

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class} ({confidence_score:.2%})")
    plt.axis("off")
    plt.show()

    return predicted_class, confidence_score

def main():
    device = get_device()
    model, class_names = load_model(device)
    transform = get_transform()

    image_path = input("Enter image path: ").strip()

    predict_image(image_path, model, class_names, transform, device)

if __name__ == "__main__":
    main()