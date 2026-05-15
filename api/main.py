import base64
import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

# api/main.py: loads model path/name/dropout from env vars and handles wrapped state_dict.
# api/download_model.py: checks models/best_model.pt; if missing, downloads from GCS.
# docker/Dockerfile.api: runs downloader first, then starts FastAPI with Uvicorn.
# .dockerignore: keeps data, venvs, secrets, and model binaries out of the Docker build context.

# Ensure project root is visible when running from /app or /app/api.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_val_transform
from src.models.model import XRayClassifier

# Step 1: Rebuild architecture
model = XRayClassifier(
    os.environ.get("MODEL_NAME", "efficientnet_b0"),
    dropout=float(os.environ.get("MODEL_DROPOUT", "0.115")),
    pretrained=False
)

# Step 2: Load weights
model_path = Path(os.environ.get("MODEL_LOCAL_PATH", PROJECT_ROOT / "models" / "best_model.pt"))
state_dict = torch.load(model_path, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
model.eval()

# Step 3: Define transforms
transform = get_val_transform()

# Step 4: FastAPI app
app = FastAPI()

# added Grad-CAM support to api/main.py (line 47): /predict now returns the usual prediction and probability, plus:
# gradcam_overlay_png: base64 PNG overlay on the X-ray
# gradcam_heatmap_png: base64 PNG heatmap

def image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def find_last_conv_layer(module: nn.Module) -> nn.Module:
    last_conv_layer = None
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            last_conv_layer = layer

    if last_conv_layer is None:
        raise RuntimeError("Could not find a convolutional layer for Grad-CAM.")

    return last_conv_layer

def create_gradcam(image: Image.Image, tensor: torch.Tensor) -> tuple[float, Image.Image, Image.Image]:
    target_layer = find_last_conv_layer(model.backbone)
    activations = None
    gradients = None

    def forward_hook(_module, _inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(_module, _grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        abnormal_logit = output[0]
        abnormal_logit.backward()
        probability = torch.sigmoid(abnormal_logit).item()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

    pooled_gradients = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (pooled_gradients * activations).sum(dim=1).squeeze(0)
    cam = torch.relu(cam)

    if torch.max(cam) > 0:
        cam = cam / torch.max(cam)

    cam_array = cam.detach().cpu().numpy()
    cam_image = Image.fromarray(np.uint8(cam_array * 255), mode="L").resize((224, 224), Image.Resampling.BILINEAR)
    heatmap = np.asarray(cam_image, dtype=np.float32) / 255.0

    base_image = image.resize((224, 224), Image.Resampling.BILINEAR).convert("RGB")
    base_array = np.asarray(base_image, dtype=np.float32)

    heat_color = np.array([255.0, 64.0, 0.0], dtype=np.float32)
    alpha = (0.45 * heatmap)[..., None]
    overlay_array = (base_array * (1.0 - alpha)) + (heat_color * alpha)
    overlay = Image.fromarray(np.uint8(np.clip(overlay_array, 0, 255)), mode="RGB")

    heatmap_rgb = Image.fromarray(np.uint8(heatmap * 255), mode="L").convert("RGB")
    return probability, overlay, heatmap_rgb


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    prob, overlay, heatmap = create_gradcam(image, tensor)

    result = "Abnormal" if prob >= 0.5 else "Normal"
    return {
        "prediction": result,
        "probability": prob,
        "gradcam_overlay_png": image_to_base64_png(overlay),
        "gradcam_heatmap_png": image_to_base64_png(heatmap),
    }

# local
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs

# with docker
# docker build -f docker/Dockerfile.api -t xray-api .
# docker run -p 8000:8000 xray-api

# docker run -p 8000:8000 -v "${PWD}:/app" -e GOOGLE_APPLICATION_CREDENTIALS=/app/project-a5834adc-8f24-44e6-8f1-0647c19fa23b.json xray-api
