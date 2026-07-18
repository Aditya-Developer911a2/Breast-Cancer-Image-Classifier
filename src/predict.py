"""
Model loading and inference. Supports two ways to get the trained model
onto the deployment machine:

1. A local file committed via Git LFS (model/<filename>.keras)
2. A download from a Hugging Face Hub model repo at app startup

Local file is tried first. This keeps local development simple (just
drop the file in model/) while still supporting a clean, repo-size-friendly
deployment via Hugging Face Hub — see README.md for which one you're using
and how to configure it.
"""

import os
import numpy as np
import tensorflow as tf
import streamlit as st

from src.preprocess import load_and_preprocess, format_prediction

# ---- Configuration ----
# Update these to match whichever checkpoint you're actually deploying.
BACKBONE = "EfficientNetB0"  # or "ResNet50" — must match the loaded model
LOCAL_MODEL_PATH = "model/efficientnetb0_seed42.keras"

# Only used if the local file isn't found. Leave HF_REPO_ID as None to
# disable this path entirely and fail with a clear error instead.
HF_REPO_ID = os.environ.get("HF_MODEL_REPO", None)  # e.g. "yourname/breakhis-effnetb0"
HF_FILENAME = os.environ.get("HF_MODEL_FILENAME", "efficientnetb0_seed42.keras")


@st.cache_resource(show_spinner="Loading model (first request only)...")
def load_model():
    """
    Loads the trained model once per app session and caches it —
    without this, Streamlit would reload the model from disk on every
    single interaction, which is slow and wasteful.
    """
    if os.path.exists(LOCAL_MODEL_PATH):
        return tf.keras.models.load_model(LOCAL_MODEL_PATH)

    if HF_REPO_ID:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        return tf.keras.models.load_model(model_path)

    raise FileNotFoundError(
        f"No model found at '{LOCAL_MODEL_PATH}' and HF_MODEL_REPO is not set. "
        "Either commit a model file at that path (via Git LFS) or set the "
        "HF_MODEL_REPO environment variable / Streamlit secret to download "
        "from Hugging Face Hub. See README.md for setup instructions."
    )


def predict(pil_image, threshold: float = 0.5) -> dict:
    """End-to-end: raw uploaded image -> structured prediction result."""
    model = load_model()
    batch = load_and_preprocess(pil_image, backbone=BACKBONE)
    raw_prob = float(model.predict(batch, verbose=0)[0][0])
    return format_prediction(raw_prob, threshold=threshold)
