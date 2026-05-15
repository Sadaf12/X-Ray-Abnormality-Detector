import os
from pathlib import Path

from google.cloud import storage


def download_model_if_missing():
    bucket_name = os.environ.get("DATA_BUCKET", "nih-xray-data")
    gcs_path = os.environ.get("MODEL_GCS_PATH", "models/best_model.pt")
    local_path = Path(os.environ.get("MODEL_LOCAL_PATH", "models/best_model.pt"))

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"Model weights already present at {local_path}")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading gs://{bucket_name}/{gcs_path} to {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    if not blob.exists():
        raise FileNotFoundError(f"Model weights not found: gs://{bucket_name}/{gcs_path}")

    blob.download_to_filename(local_path)
    print(f"Downloaded model weights to {local_path}")


if __name__ == "__main__":
    download_model_if_missing()
