from torchvision import models
import torch.nn as nn

def PlantDiseaseModel(num_classes: int, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    model = model.to(device)

    print(f"Model final layer: {model.fc}")
    return model