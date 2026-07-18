"""
Preprocessing utilities — mirrors the exact pipeline used during training
(see the project notebooks), so inference-time preprocessing matches what
the model was actually trained on. Getting this wrong is the single most
common reason a deployed model performs worse than its reported metrics.
"""

import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = (224, 224)

# Must match training exactly — each backbone expects pixels scaled
# differently, since each was pretrained with a specific convention.
PREPROCESS_FNS = {
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "ResNet50": tf.keras.applications.resnet.preprocess_input,
}

LABEL_NAMES = {0: "Benign", 1: "Malignant"}


def load_and_preprocess(pil_image: Image.Image, backbone: str) -> np.ndarray:
    """
    Convert a PIL image (as received from Streamlit's file uploader) into
    a model-ready batch of shape (1, 224, 224, 3).

    Steps mirror training: RGB conversion, resize to 224x224, cast to
    float32, then backbone-specific normalization. No augmentation is
    applied here — augmentation is a training-time-only technique;
    applying it at inference would just add unnecessary noise to a
    single prediction.
    """
    if backbone not in PREPROCESS_FNS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Expected one of {list(PREPROCESS_FNS)}."
        )

    img = pil_image.convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.BILINEAR)

    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    preprocess_fn = PREPROCESS_FNS[backbone]
    arr = preprocess_fn(arr)

    return arr


def format_prediction(raw_probability: float, threshold: float = 0.5) -> dict:
    """
    Convert the model's raw sigmoid output (P(malignant)) into a
    structured result. Threshold matches training/evaluation default;
    change here if you later tune the operating point (e.g. lowering it
    to trade specificity for sensitivity, as discussed in the project's
    own results — sensitivity consistently lagged specificity).
    """
    predicted_label = 1 if raw_probability >= threshold else 0
    confidence = raw_probability if predicted_label == 1 else (1 - raw_probability)

    return {
        "label": LABEL_NAMES[predicted_label],
        "label_id": predicted_label,
        "p_malignant": float(raw_probability),
        "p_benign": float(1 - raw_probability),
        "confidence": float(confidence),
    }
