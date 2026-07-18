"""
BreaKHis Breast Cancer Histopathology Classifier — Streamlit demo app.

Research/educational demo only — see the "Important Limitations" section
in the app and in README.md before drawing any conclusions from it.
"""

import streamlit as st
from PIL import Image

from src.predict import predict, BACKBONE

st.set_page_config(
    page_title="Breast Cancer Histopathology Classifier",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Breast Cancer Histopathology Classifier")
st.caption(
    f"Transfer-learning classifier ({BACKBONE}) trained on the BreaKHis dataset "
    "— benign vs. malignant tissue classification from histopathology images."
)

with st.expander("⚠️ Important limitations — read before using", expanded=False):
    st.markdown(
        """
This is a **research and educational demo**, not a diagnostic tool.

- **Not for clinical use.** This model has not been clinically validated
  and must never be used to inform an actual diagnosis or treatment
  decision.
- **Small test set.** Reported metrics come from a patient-level test
  set of only ~12 patients — a genuinely strong result on a small
  sample, not proof of clinical-grade reliability.
- **Trained on one dataset.** BreaKHis images come from a specific
  staining and imaging protocol; performance on images from a different
  lab, scanner, or preparation method is unknown and may be substantially
  worse.
- **Single model, single seed.** This demo runs one trained checkpoint.
  The full project evaluated multiple architectures across five random
  seeds — see the README for the complete, honest picture including
  seed-to-seed variance.
        """
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload a histopathology image (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running inference..."):
        try:
            result = predict(image)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    with col2:
        st.subheader("Prediction")

        if result["label_id"] == 1:
            st.error(f"**{result['label']}**")
        else:
            st.success(f"**{result['label']}**")

        st.metric("Confidence", f"{result['confidence']*100:.1f}%")

        st.write("Class probabilities:")
        st.progress(result["p_malignant"], text=f"Malignant: {result['p_malignant']*100:.1f}%")
        st.progress(result["p_benign"], text=f"Benign: {result['p_benign']*100:.1f}%")

    st.divider()
    st.caption(
        "Remember: this is a demo of a research project, not a medical device. "
        "See the limitations above."
    )
else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.caption(
    "Built as part of a research internship comparing Baseline CNN, ResNet50, "
    "and EfficientNetB0 on the BreaKHis dataset, with a five-seed reproducibility "
    "study. Full methodology and results in the project README."
)
