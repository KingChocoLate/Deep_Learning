import torchvision
import torch
from torch import nn
from torchvision import models

def Multilabel_Model(num_classes):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        input_features,
        num_classes
    )
    return model