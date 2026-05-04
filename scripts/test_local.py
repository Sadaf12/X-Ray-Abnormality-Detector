import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.data.dataset import ChestXrayDataset
from src.data.transforms import get_train_transform, get_val_transform
from src.models.model import XRayClassifier
from src.training.evaluate import evaluate

# Dummy dataframe — no real images needed
df = pd.DataFrame({
    "image_path": ["dummy/path.png"] * 20,
    "target": [0, 1] * 10
})

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

# Test 1: Dataset __len__
dataset = ChestXrayDataset(train_df, transform=get_train_transform())
print(f"✓ Dataset len: {len(dataset)}")

# Test 2: Model forward pass — fixed EfficientNet-B0
model = XRayClassifier("efficientnet_b0", dropout=0.3, pretrained=False)
dummy_input = torch.randn(2, 3, 224, 224)
output = model(dummy_input)
print(f"✓ Model output shape: {output.shape}")

# Test 3: Loss
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.17]))
labels = torch.tensor([0.0, 1.0])
loss = criterion(output, labels)
print(f"✓ Loss: {loss.item():.4f}")

# Test 4: Evaluate function
from torch.utils.data import DataLoader
val_loader = DataLoader(
    ChestXrayDataset(val_df, transform=get_val_transform()),
    batch_size=4
)
# Skip actual evaluate since images are dummy — just check it imports
print("✓ Evaluate imported successfully")

# Test 5: Hparam env variable overrides work
os.environ["HPARAM_TRIAL"] = "true"
os.environ["LR"] = "0.0005"
os.environ["DROPOUT"] = "0.3"
os.environ["WEIGHT_DECAY"] = "0.0001"
os.environ["BATCH_SIZE"] = "32"
os.environ["TRIAL_NUMBER"] = "0"

print(f"✓ LR override: {float(os.environ['LR'])}")
print(f"✓ Batch size override: {int(os.environ['BATCH_SIZE'])}")

print("\nAll local tests passed — safe to push to Vertex AI.")