import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch

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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    result = "Abnormal" if prob >= 0.5 else "Normal"
    return {"prediction": result, "probability": prob}

# local
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs

# with docker
# docker build -f docker/Dockerfile.api -t xray-api .
# docker run -p 8000:8000 xray-api

# docker run -p 8000:8000 -v "${PWD}:/app" -e GOOGLE_APPLICATION_CREDENTIALS=/app/project-a5834adc-8f24-44e6-8f1-0647c19fa23b.json xray-api