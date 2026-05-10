# This script launches training on Vertex AI.
# Use this for production runs or when VM is not available.
# For local VM training, run src/training/train.py directly.

import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID     = os.environ["PROJECT_ID"]
REGION         = "europe-west4"
IMAGE_URI      = f"europe-west4-docker.pkg.dev/{PROJECT_ID}/xray-abnormality-detector-eu/xray-train:latest"
DATA_BUCKET    = os.environ["DATA_BUCKET"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
WANDB_API_KEY  = os.environ["WANDB_API_KEY"]

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{STAGING_BUCKET}",
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="xray-hparam-search",
    container_uri=IMAGE_URI,
)

job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    boot_disk_size_gb=100,
    environment_variables={
        "WANDB_API_KEY": WANDB_API_KEY,
        "DATA_BUCKET":   DATA_BUCKET,
        "RUN_MODE":      "hparam",
        "N_TRIALS":      "10",
    },
    replica_count=1,
    sync=False,
)

print("Hyperparameter search job submitted — you can close your terminal.")
print("Monitor at: https://wandb.ai/sadafambreen-zhaw/X-Ray-Abnormality-Detector")