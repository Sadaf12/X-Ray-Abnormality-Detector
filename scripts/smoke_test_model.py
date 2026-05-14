import argparse
import os
import sys
from pathlib import Path

import torch
from google.cloud import storage

# Google Cloud authentication works
# GCS model exists
# model downloads correctly
# architecture reconstructs correctly
# weights load correctly
# inference runs
# output shape correct
# no NaN/inf values

sys.path.insert(0, os.getcwd())

from src.models.model import XRayClassifier


def download_weights(bucket_name, gcs_path, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading gs://{bucket_name}/{gcs_path} to {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise FileNotFoundError(f"Model weights not found: gs://{bucket_name}/{gcs_path}")

    blob.download_to_filename(local_path)
    return local_path


def load_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def run_smoke_test(args):
    weights_path = download_weights(args.bucket, args.gcs_path, args.local_path)

    model = XRayClassifier(
        model_name=args.model_name,
        dropout=args.dropout,
        pretrained=False,
    )
    model.load_state_dict(load_state_dict(weights_path))
    model.eval()

    dummy_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    expected_shape = (args.batch_size,)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}")
    if not torch.isfinite(probs).all():
        raise AssertionError("Model produced non-finite probabilities")

    print(f"Smoke test passed: logits shape={tuple(logits.shape)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download model weights from GCS and run a quick inference smoke test.")
    parser.add_argument("--bucket", default=os.environ.get("DATA_BUCKET", "nih-xray-data"))
    parser.add_argument("--gcs-path", default=os.environ.get("MODEL_GCS_PATH", "models/best_model.pt"))
    parser.add_argument("--local-path", default=os.environ.get("MODEL_LOCAL_PATH", "models/best_model.pt"))
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "efficientnet_b0"))
    parser.add_argument("--dropout", type=float, default=float(os.environ.get("MODEL_DROPOUT", "0.115")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("SMOKE_BATCH_SIZE", "2")))
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("SMOKE_IMAGE_SIZE", "224")))
    return parser.parse_args()


if __name__ == "__main__":
    run_smoke_test(parse_args())
