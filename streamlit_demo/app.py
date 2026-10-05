from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = (
    PROJECT_DIR
    / "saved_models"
    / "multilabel_model_pos_weights_added.pth"
)

# This order MUST be identical to the order used when creating your
# multi-hot targets during training.
DEFAULT_LABEL_NAMES = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]

DISPLAY_NAMES = {
    "complex": "Complex",
    "frog_eye_leaf_spot": "Frog-eye leaf spot",
    "healthy": "Healthy",
    "powdery_mildew": "Powdery mildew",
    "rust": "Rust",
    "scab": "Scab",
}

DEFAULT_THRESHOLD = 0.50
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png", "webp"]


# ---------------------------------------------------------------------------
# Page setup and visual styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Apple Leaf Health Check",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .app-subtitle {
            color: #5f6b63;
            font-size: 1.02rem;
            margin-top: -0.6rem;
            margin-bottom: 1.8rem;
        }

        .result-card {
            border: 1px solid #e3e8e4;
            border-radius: 14px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            background: #ffffff;
        }

        .result-card.healthy {
            border-left: 5px solid #3b7d4b;
            background: #f5faf6;
        }

        .result-card.warning {
            border-left: 5px solid #b87914;
            background: #fffaf1;
        }

        .result-card.uncertain {
            border-left: 5px solid #7a8288;
            background: #f7f8f8;
        }

        .result-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1f2a23;
            margin-bottom: 0.25rem;
        }

        .result-description {
            color: #536059;
            line-height: 1.5;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.8rem;
        }

        .label-chip {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #edf3ee;
            color: #294d31;
            border: 1px solid #d4e3d7;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .probability-list {
            margin-top: 0.5rem;
        }

        .probability-row {
            margin-bottom: 0.9rem;
        }

        .probability-heading {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 1rem;
            margin-bottom: 0.3rem;
        }

        .probability-name {
            font-weight: 600;
            color: #26322b;
        }

        .probability-value {
            font-variant-numeric: tabular-nums;
            color: #526059;
        }

        .probability-track {
            width: 100%;
            height: 10px;
            background: #e9eeea;
            border-radius: 999px;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            border-radius: 999px;
            background: #4f7658;
        }

        .threshold-line {
            color: #68736c;
            font-size: 0.86rem;
            margin-top: 0.25rem;
        }

        .empty-state {
            border: 1px dashed #cdd5cf;
            border-radius: 14px;
            padding: 2.2rem 1.4rem;
            text-align: center;
            color: #637068;
            background: #fafcfb;
        }

        .small-note {
            color: #68736c;
            font-size: 0.88rem;
            line-height: 1.45;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model-loading helpers
# ---------------------------------------------------------------------------

def safe_torch_load(path: Path, device: torch.device) -> Any:
    """
    Load a state-dict checkpoint safely.

    weights_only=True is supported by current PyTorch releases. The TypeError
    fallback only exists for older PyTorch versions that do not expose this
    argument.
    """
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """
    Support either:
      1. torch.save(model.state_dict(), path)
      2. torch.save({"model_state_dict": model.state_dict(), ...}, path)
      3. torch.save({"state_dict": model.state_dict(), ...}, path)
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "The model file must contain a state_dict or a checkpoint "
            "dictionary containing 'model_state_dict' or 'state_dict'."
        )

    # A plain state_dict is a dictionary whose values are tensors.
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    for key in ("model_state_dict", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    raise ValueError(
        "No model state_dict was found. Expected a plain state_dict or a "
        "checkpoint containing 'model_state_dict' or 'state_dict'."
    )


def remove_training_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Remove prefixes sometimes added by DataParallel or torch.compile.
    """
    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ("module.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def read_checkpoint_metadata(
    checkpoint: Any,
) -> tuple[list[str], float]:
    """
    Read optional label names and a scalar threshold from the checkpoint.

    If they were not saved, use the constants declared above.
    """
    label_names = DEFAULT_LABEL_NAMES.copy()
    threshold = DEFAULT_THRESHOLD

    if isinstance(checkpoint, dict):
        saved_labels = (
            checkpoint.get("label_names")
            or checkpoint.get("class_names")
            or checkpoint.get("labels")
        )

        if isinstance(saved_labels, (list, tuple)):
            label_names = [str(label) for label in saved_labels]

        saved_threshold = checkpoint.get("threshold")
        if isinstance(saved_threshold, (int, float)):
            threshold = float(saved_threshold)

    if len(label_names) != len(DEFAULT_LABEL_NAMES):
        raise ValueError(
            f"The checkpoint contains {len(label_names)} labels, but this app "
            f"expects {len(DEFAULT_LABEL_NAMES)} outputs."
        )

    return label_names, threshold


@st.cache_resource(show_spinner="Loading the trained model...")
def load_model(
    model_path: str,
) -> tuple[
    nn.Module,
    torch.device,
    list[str],
    Any,
    float,
]:
    """
    Build EfficientNet-B4, restore its trained parameters, and cache it so the
    model is not reloaded after every Streamlit interaction.
    """
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {checkpoint_path}\n\n"
            "Place your checkpoint at saved_models/best_model.pth, or update "
            "MODEL_PATH near the top of app.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = safe_torch_load(checkpoint_path, device)
    label_names, saved_threshold = read_checkpoint_metadata(checkpoint)
    state_dict = remove_training_prefixes(extract_state_dict(checkpoint))

    # weights=None avoids downloading ImageNet parameters during deployment.
    # Your own trained state_dict is loaded immediately below.
    model = efficientnet_b4(weights=None)

    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        input_features,
        len(label_names),
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint does not match an EfficientNet-B4 model with "
            f"{len(label_names)} outputs. Confirm that this app uses the same "
            "architecture and label order as training.\n\n"
            f"Original loading error:\n{error}"
        ) from error

    model.to(device)
    model.eval()

    # This transform prepares validation/test images in the same format
    # associated with EfficientNet-B4's ImageNet weight preset.
    evaluation_transform = EfficientNet_B4_Weights.DEFAULT.transforms()

    return (
        model,
        device,
        label_names,
        evaluation_transform,
        saved_threshold,
    )


# ---------------------------------------------------------------------------
# Prediction and presentation helpers
# ---------------------------------------------------------------------------

def predict_image(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    transform: Any,
    label_names: list[str],
) -> dict[str, float]:
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0).cpu()

    return {
        label_name: float(probability)
        for label_name, probability in zip(label_names, probabilities)
    }


def display_name(label: str) -> str:
    return DISPLAY_NAMES.get(
        label,
        label.replace("_", " ").title(),
    )


def render_summary(
    probabilities: dict[str, float],
    threshold: float,
) -> None:
    detected = sorted(
        [
            (label, probability)
            for label, probability in probabilities.items()
            if probability >= threshold
        ],
        key=lambda item: item[1],
        reverse=True,
    )

    disease_predictions = [
        item for item in detected if item[0] != "healthy"
    ]
    healthy_probability = probabilities.get("healthy", 0.0)

    if disease_predictions:
        card_class = "warning"
        title = "Potential disease signs detected"
        description = (
            "One or more disease labels crossed the selected threshold. "
            "Review the individual scores below."
        )
        chips = disease_predictions

    elif healthy_probability >= threshold:
        card_class = "healthy"
        title = "Healthy leaf predicted"
        description = (
            "No disease label crossed the selected threshold, while the "
            "healthy score did."
        )
        chips = [("healthy", healthy_probability)]

    else:
        card_class = "uncertain"
        title = "Result is uncertain"
        description = (
            "No label crossed the selected threshold. Try a clearer image or "
            "review the highest probability below."
        )
        chips = []

    chip_html = ""

    if chips:
        chip_html = '<div class="chip-row">' + "".join(
            (
                '<span class="label-chip">'
                f"{html.escape(display_name(label))}: {probability:.1%}"
                "</span>"
            )
            for label, probability in chips
        ) + "</div>"

    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-title">{html.escape(title)}</div>
            <div class="result-description">
                {html.escape(description)}
            </div>
            {chip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(
    probabilities: dict[str, float],
    threshold: float,
) -> None:
    sorted_probabilities = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    rows: list[str] = []

    for label, probability in sorted_probabilities:
        bounded_probability = min(max(probability, 0.0), 1.0)
        percentage = bounded_probability * 100

        rows.append(
            f"""
            <div class="probability-row">
                <div class="probability-heading">
                    <span class="probability-name">
                        {html.escape(display_name(label))}
                    </span>
                    <span class="probability-value">
                        {percentage:.1f}%
                    </span>
                </div>
                <div class="probability-track">
                    <div
                        class="probability-fill"
                        style="width: {percentage:.2f}%;">
                    </div>
                </div>
            </div>
            """
        )

    st.markdown(
        '<div class="probability-list">'
        + "".join(rows)
        + "</div>"
        + (
            '<div class="threshold-line">'
            f"Current decision threshold: {threshold:.2f}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

st.title("Apple Leaf Health Check")
st.markdown(
    """
    <div class="app-subtitle">
        Upload or photograph an apple leaf. The model evaluates all six labels
        independently and shows the probability for each one.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    (
        model,
        device,
        label_names,
        evaluation_transform,
        checkpoint_threshold,
    ) = load_model(str(MODEL_PATH))
except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.stop()


with st.sidebar:
    st.header("Prediction settings")

    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(
            min(max(checkpoint_threshold, 0.10), 0.90)
        ),
        step=0.01,
        help=(
            "A label is reported as detected when its probability is equal to "
            "or above this value. Start with 0.50 unless you selected another "
            "threshold using validation data."
        ),
    )

    st.divider()

    st.subheader("Model")
    st.write("EfficientNet-B4")
    st.caption(f"Running on: {str(device).upper()}")

    st.divider()

    st.markdown(
        """
        <div class="small-note">
            Use a clear image with the leaf in focus. Avoid heavy blur,
            extreme shadows, or a leaf that occupies only a tiny part of the
            frame.
        </div>
        """,
        unsafe_allow_html=True,
    )


input_method = st.radio(
    "Choose image source",
    options=("Upload an image", "Use camera"),
    horizontal=True,
    label_visibility="collapsed",
)

image_source = None

if input_method == "Upload an image":
    image_source = st.file_uploader(
        "Upload an apple-leaf image",
        type=SUPPORTED_FILE_TYPES,
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
    )
else:
    image_source = st.camera_input(
        "Take a clear photo of the apple leaf",
    )


if image_source is None:
    st.markdown(
        """
        <div class="empty-state">
            Add an image to begin. The app will automatically evaluate
            complex, frog-eye leaf spot, healthy, powdery mildew, rust,
            and scab.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


try:
    image = Image.open(image_source).convert("RGB")
except (UnidentifiedImageError, OSError) as error:
    st.error("The selected file could not be opened as an image.")
    st.code(str(error))
    st.stop()


with st.spinner("Analysing the leaf..."):
    probabilities = predict_image(
        image=image,
        model=model,
        device=device,
        transform=evaluation_transform,
        label_names=label_names,
    )


image_column, result_column = st.columns(
    [1.05, 1.25],
    gap="large",
)

with image_column:
    st.subheader("Input image")
    st.image(
        image,
        use_container_width=True,
    )

with result_column:
    st.subheader("Prediction")
    render_summary(
        probabilities=probabilities,
        threshold=threshold,
    )

    top_label, top_probability = max(
        probabilities.items(),
        key=lambda item: item[1],
    )

    metric_left, metric_right = st.columns(2)

    with metric_left:
        st.metric(
            "Highest-scoring label",
            display_name(top_label),
        )

    with metric_right:
        st.metric(
            "Highest probability",
            f"{top_probability:.1%}",
        )


st.divider()
st.subheader("All label probabilities")
render_probability_bars(
    probabilities=probabilities,
    threshold=threshold,
)

with st.expander("Technical details"):
    st.write(
        {
            "architecture": "EfficientNet-B4",
            "number_of_outputs": len(label_names),
            "label_order": label_names,
            "device": str(device),
            "decision_threshold": threshold,
            "input_mode": input_method,
        }
    )

st.caption(
    "This demo reports model predictions for educational testing. "
    "It should not replace confirmation by a plant-health specialist."
)