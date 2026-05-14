import os, io, sys
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms

# Ensure src folder visible
sys.path.insert(0, os.getcwd())

# Import architecture
from src.models.model import XRayClassifier

# Step 1: Rebuild architecture
model = XRayClassifier("efficientnet_b0", dropout=0.3, pretrained=False)

# Step 2: Load weights
state_dict = torch.load(os.path.join("models", "best_model.pt"), map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Step 3: Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

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

    result = "Abnormal" if prob > 0.5 else "Normal"
    return {"prediction": result, "probability": prob}



#cd "E:\4 machine learning data operation thu mr\X-Ray-Abnormality-Detector"
#uvicorn main:app --reload --host 0.0.0.0 --port 8000
#http://127.0.0.1:8000/docs

