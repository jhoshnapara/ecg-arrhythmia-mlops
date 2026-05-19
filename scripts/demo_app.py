"""
Streamlit demo for ECG arrhythmia classification.

This is the public-facing demo. Drop it on Hugging Face Spaces and you have
a live URL to put on your resume.

What it does:
  - Lets the user pick a sample beat from the test set, or upload a CSV
  - Plots the ECG waveform
  - Calls the model and shows the predicted class + per-class probabilities
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.cnn import ECGCNN

CLASS_NAMES = ["Normal (N)", "Supraventricular (S)", "Ventricular (V)", "Fusion (F)", "Unclassified (Q)"]
CLASS_DESCRIPTIONS = {
    "Normal (N)": "Normal sinus rhythm beat.",
    "Supraventricular (S)": "Atrial or junctional premature beat.",
    "Ventricular (V)": "Premature ventricular contraction or escape beat. Clinically important — high recall priority.",
    "Fusion (F)": "Fusion of ventricular and normal beat.",
    "Unclassified (Q)": "Paced or otherwise unclassifiable beat.",
}

st.set_page_config(page_title="ECG Arrhythmia Classifier", layout="wide")
st.title("🫀 ECG Arrhythmia Classifier")
st.markdown(
    "1D CNN trained on MIT-BIH Arrhythmia Database. "
    "Classifies 1-second ECG windows into 5 AAMI classes."
)


@st.cache_resource
def load_model():
    """Load model weights. In production this comes from MLflow registry."""
    model = ECGCNN()
    weights_path = Path("models/best.pt")
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        st.warning("⚠️ No trained weights found at `models/best.pt`. Demo running with random init.")
    model.eval()
    return model


@st.cache_data
def load_test_samples():
    """Cache the test set so picking samples is instant."""
    data = np.load("data/processed/test.npz")
    return data["X"], data["y"]


model = load_model()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Pick a beat")
    source = st.radio("Source", ["Test set sample", "Upload CSV (360 samples)"])

    if source == "Test set sample":
        try:
            X_test, y_test = load_test_samples()
            class_filter = st.selectbox("Filter by ground truth class", ["All"] + CLASS_NAMES)
            if class_filter == "All":
                candidate_idxs = list(range(len(X_test)))
            else:
                class_idx = CLASS_NAMES.index(class_filter)
                candidate_idxs = np.where(y_test == class_idx)[0].tolist()

            idx = st.slider("Beat index", 0, len(candidate_idxs) - 1, 0)
            beat_idx = candidate_idxs[idx]
            signal = X_test[beat_idx]
            true_label = CLASS_NAMES[int(y_test[beat_idx])]
        except FileNotFoundError:
            st.error("Run `python scripts/preprocess.py` first.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload single-beat CSV (one column, 360 rows)", type="csv")
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        signal = np.loadtxt(uploaded, delimiter=",").astype(np.float32)
        if len(signal) != 360:
            st.error(f"Expected 360 samples, got {len(signal)}.")
            st.stop()
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        true_label = None

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode="lines", line=dict(color="#dc2626", width=2), name="ECG"))
    fig.update_layout(
        xaxis_title="Sample (360 Hz)",
        yaxis_title="Amplitude (z-scored)",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Prediction")
    with torch.no_grad():
        logits = model(torch.from_numpy(signal).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    st.metric("Predicted class", pred_class)
    st.metric("Confidence", f"{probs[pred_idx]*100:.1f}%")
    if true_label:
        match = "✅" if true_label == pred_class else "❌"
        st.metric(f"Ground truth {match}", true_label)

    st.markdown("**Class probabilities:**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{name}: {p*100:.1f}%")

    st.caption(CLASS_DESCRIPTIONS[pred_class])

st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown("""
    - **Architecture:** 3-block 1D CNN (~80K params)
    - **Training data:** MIT-BIH Arrhythmia Database (DS1 split)
    - **Input:** 1-second ECG window @ 360 Hz, z-score normalized
    - **Disclaimer:** Demo only. Not for clinical decision-making.
    """)
