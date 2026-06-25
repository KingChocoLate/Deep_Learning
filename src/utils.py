import numpy as np
import matplotlib.pyplot as plt
import torch

def show_image(image_tensor, label, class_names):
    image = image_tensor.permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title(class_names[label])
    plt.axis(False)
    plt.show()

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return device