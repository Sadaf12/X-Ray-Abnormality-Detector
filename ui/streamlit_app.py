import base64
import io
import os

from PIL import Image
import requests
import streamlit as st


API_URL = os.environ.get("XRAY_API_URL", "http://127.0.0.1:8000/predict")
GRADCAM_KEYS = ("gradcam_overlay_png", "gradcam_heatmap_png")


st.set_page_config(
    page_title="X-Ray Abnormality Detector",
    layout="centered",
)

st.title("X-Ray Abnormality Detector")

with st.expander("Information"):
    st.write(
        """
        This website was created during the MLOps module at the MSE program.
        The main focus of the project was the development of a lightweight
        end-to-end machine learning pipeline rather than maximizing model performance.

        Users can upload chest X-ray images and receive a prediction indicating
        whether the image is classified as normal or abnormal.

        This project is intended for educational and demonstration purposes only
        and is not a medical diagnostic tool.
        """
    )

st.caption("Upload a chest X-ray image to classify it and visualize Grad-CAM regions.")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
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

    col1, col2 = st.columns(2)

    with col1:
        if prediction == "Abnormal":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

    with col2:
        st.metric("Abnormal probability", f"{probability:.1%}")

    st.divider()

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
        st.warning("The API responded without Grad-CAM images.")
        st.image(original_image, caption="Original image", width=450)
        with st.expander("API response"):
            st.json(result)