from pathlib import Path
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from utils import get_device, show_image


# Allow app.py to import files from src/
sys.path.append("src")

from model import PlantDiseaseModel


MODEL_PATH = Path("saved_models/plant_disease_resnet18_model.pth")


st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="centered"
)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


@st.cache_resource
def load_model():
    device = get_device()

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = PlantDiseaseModel(
        num_classes=num_classes,
        device=device
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def predict_image(image, model, class_names, device):
    transform = get_transform()

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output_logits = model(image_tensor)
        probabilities = torch.softmax(output_logits, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]
    confidence_score = confidence.item()

    return predicted_class, confidence_score, probabilities.squeeze().cpu()


st.title("🍅 Tomato Leaf Disease Classifier")

st.write(
    "Upload a tomato leaf image, and the model will predict the possible disease class."
)

st.caption(
    "This is a learning demo built with PyTorch, ResNet18 transfer learning, and Streamlit."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH}`\n\n"
        "Please train your model first using `python src/train.py`."
    )
    st.stop()


model, class_names, device = load_model()

st.info(f"Model loaded successfully. Device: `{device}`")

uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Disease"):
        predicted_class, confidence_score, probabilities = predict_image(
            image=image,
            model=model,
            class_names=class_names,
            device=device
        )

        st.success(f"Prediction: **{predicted_class}**")
        st.write(f"Confidence: **{confidence_score:.2%}**")

        st.subheader("Class Probabilities")

        probability_data = {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }

        st.bar_chart(probability_data)

        st.warning(
            "This project is for learning and demonstration only. "
            "It should not be used as a real agricultural diagnosis system yet."
        )