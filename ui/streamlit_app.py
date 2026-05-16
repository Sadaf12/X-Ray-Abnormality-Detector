import base64
import io
import json
import os

import matplotlib.pyplot as plt
from google.cloud import storage
from PIL import Image
import requests
import streamlit as st


# Read runtime configuration from environment variables.
API_URL = os.environ.get("XRAY_API_URL", "http://127.0.0.1:8000/predict")
MODEL_INFO_BUCKET = os.environ.get("MODEL_INFO_BUCKET", os.environ.get("DATA_BUCKET", "nih-xray-data"))
METRICS_GCS_PATH = os.environ.get("METRICS_GCS_PATH", "models/evaluation/test_metrics.json")
ROC_GCS_PATH = os.environ.get("ROC_GCS_PATH", "models/evaluation/roc_curve.json")
GRADCAM_KEYS = ("gradcam_overlay_png", "gradcam_heatmap_png")

# Fallback values keep the UI usable when GCS metrics are unavailable.
DEFAULT_METRICS = {
    "training_data": 89696,
    "validation_data": 11212,
    "test_data": 11212,
    "total_data": 112120,
    "target_distribution": {
        "0": 0.538352,
        "1": 0.461648,
    },
    "test_loss": 0.6170361638069153,
    "test_accuracy": 0.7221726721369961,
    "test_auc": 0.7836486682685955,
    "threshold": 0.4995993,
    "threshold_label": 0.5,
    "tpr": 0.6810278207109737,
    "fpr": 0.24271040424121934,
    "best_test_accuracy": 0.7221726721369961,
}
DEFAULT_ROC = {
    "fpr": [0.0, 0.02, 0.05, 0.10, 0.18, 0.25, 0.35, 0.50, 0.65, 0.82, 1.0],
    "tpr": [0.0, 0.12, 0.28, 0.43, 0.60, 0.70, 0.78, 0.87, 0.92, 0.97, 1.0],
}


# Configure the Streamlit browser tab and use a wide page for the side panel.
st.set_page_config(
    page_title="X-Ray Abnormality Detector",
    layout="wide",
)


@st.cache_data(ttl=300)
def load_model_info():
    """Load model evaluation artifacts from GCS, falling back to bundled values."""
    try:
        client = storage.Client()
        bucket = client.bucket(MODEL_INFO_BUCKET)
        metrics = json.loads(bucket.blob(METRICS_GCS_PATH).download_as_text())
        roc_data = json.loads(bucket.blob(ROC_GCS_PATH).download_as_text())
        return metrics, roc_data, f"Loaded from gs://{MODEL_INFO_BUCKET}/{METRICS_GCS_PATH}"
    except Exception as exc:
        return DEFAULT_METRICS, DEFAULT_ROC, f"Using bundled fallback metrics: {exc}"


def render_model_info_panel(metrics, roc_data, source_message):
    """Render the persistent right-side panel with dataset and performance info."""
    with st.container(border=True):
        st.subheader("Trained Model")
        st.caption("Held-out test evaluation with threshold 50%")

        # Show the train/validation/test split used by the model.
        st.markdown("**Dataset Split**")
        split_cols = st.columns(2)
        split_cols[0].metric("Training Data", f"{metrics['training_data']:,}")
        split_cols[1].metric("Validation Data", f"{metrics['validation_data']:,}")
        split_cols[0].metric("Test Data", f"{metrics['test_data']:,}")
        split_cols[1].metric("Total Data", f"{metrics['total_data']:,}")

        # Show the binary class balance of the dataset.
        st.markdown("**Target Distribution**")
        st.write("`0` = no abnormality, `1` = abnormality")
        target_distribution = metrics["target_distribution"]
        st.dataframe(
            {
                "target": [0, 1],
                "fraction": [
                    target_distribution.get("0", 0.0),
                    target_distribution.get("1", 0.0),
                ],
            },
            hide_index=True,
            use_container_width=True,
        )

        # Show headline metrics from the held-out test set.
        st.markdown("**Test Performance**")
        perf_cols = st.columns(3)
        perf_cols[0].metric("Loss", f"{metrics['test_loss']:.4f}")
        perf_cols[1].metric("Accuracy", f"{metrics['test_accuracy']:.2%}")
        perf_cols[2].metric("AUC", f"{metrics['test_auc']:.4f}")

        # Plot the ROC curve and mark the selected decision threshold.
        st.markdown("**ROC Curve**")
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]

        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.plot(fpr, tpr, label=f"ROC AUC = {metrics['test_auc']:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
        ax.scatter(
            [metrics["fpr"]],
            [metrics["tpr"]],
            color="#d62728",
            zorder=3,
            label="Threshold = 50%",
        )
        ax.set_title("Test Set ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Display the selected operating point for the 50% threshold.
        st.markdown("**Operating Point**")
        st.write(
            f"""
            Threshold: `{metrics['threshold']:.7f}`  
            TPR: `{metrics['tpr']:.16f}`  
            FPR: `{metrics['fpr']:.16f}`  
            Best test accuracy: `{metrics['best_test_accuracy']:.16f}`
            """
        )
        st.caption(source_message)


# Load evaluation data before rendering the layout.
metrics, roc_data, source_message = load_model_info()

# Split the page into the prediction workflow and the model information panel.
left_col, right_col = st.columns([2.2, 1], gap="large")

with left_col:
    st.title("X-Ray Abnormality Detector")

    # Explain the project context and intended educational use.
    with st.expander("Information"):
        st.write(
            """
            This website was created during the MLOps module at the MSE program.
            The main focus of the project was the development of a lightweight
            end-to-end machine learning pipeline rather than maximizing model performance.

            Users can upload chest X-ray images and receive a prediction indicating
            whether the image is classified as normal or abnormal.

            This project is intended for educational and demonstration purposes only
            and is not a medical diagnostic tool...
            """
        )

    st.caption("Upload a chest X-ray image to classify it and visualize Grad-CAM regions.")

    # Let the user upload an image that will be sent to the prediction API.
    uploaded_file = st.file_uploader(
        "Upload a chest X-ray",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        # Read the uploaded image and send the raw bytes to the FastAPI service.
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, image_bytes, uploaded_file.type)},
                    timeout=60,
                )
            except requests.RequestException as exc:
                st.error(f"Could not reach the API at {API_URL}: {exc}")
                st.stop()

        if not response.ok:
            st.error(f"API request failed: {response.status_code} {response.text}")
            st.stop()

        result = response.json()

        prediction = result["prediction"]
        probability = result["probability"]

        # Present the classification label and abnormality probability.
        col1, col2 = st.columns(2)

        with col1:
            if prediction == "Abnormal":
                st.error(f"Prediction: {prediction}")
            else:
                st.success(f"Prediction: {prediction}")

        with col2:
            st.metric("Abnormal probability", f"{probability:.1%}")

        st.divider()

        # Display Grad-CAM outputs when the API includes them.
        if all(key in result for key in GRADCAM_KEYS):
            overlay_bytes = base64.b64decode(result["gradcam_overlay_png"])
            heatmap_bytes = base64.b64decode(result["gradcam_heatmap_png"])

            tab1, tab2, tab3 = st.tabs(
                ["Original", "Grad-CAM Overlay", "Heatmap"]
            )

            with tab1:
                st.image(original_image, caption="Original image", width=450)

            with tab2:
                st.image(overlay_bytes, caption="Grad-CAM overlay", width=450)

            with tab3:
                st.image(heatmap_bytes, caption="Grad-CAM heatmap", width=450)

        else:
            # Fall back to the original image and raw API response for debugging.
            st.warning("The API responded without Grad-CAM images.")
            st.image(original_image, caption="Original image", width=450)
            with st.expander("API response"):
                st.json(result)

with right_col:
    # Keep model details visible next to the upload workflow.
    render_model_info_panel(metrics, roc_data, source_message)
