"""Launch the full Vertex AI training pipeline.

This starts one cloud job that performs:
1. hyperparameter search,
2. best-param handoff,
3. full training,
4. held-out test evaluation.
"""

import os

from dotenv import load_dotenv
from google.cloud import aiplatform


load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = "europe-west4"
IMAGE_URI = (
    f"europe-west4-docker.pkg.dev/{PROJECT_ID}/"
    "xray-abnormality-detector-eu/xray-train:latest"
)
DATA_BUCKET = os.environ["DATA_BUCKET"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
N_TRIALS = os.environ.get("N_TRIALS", "10")

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{STAGING_BUCKET}",
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="xray-full-pipeline",
    container_uri=IMAGE_URI,
)

job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    boot_disk_size_gb=100,
    environment_variables={
        "WANDB_API_KEY": WANDB_API_KEY,
        "DATA_BUCKET": DATA_BUCKET,
        "RUN_MODE": "pipeline",
        "N_TRIALS": N_TRIALS,
        "SMOKE_TEST": os.environ.get("SMOKE_TEST", "false"),
    },
    replica_count=1,
    sync=False,
)

print("Full pipeline job submitted - you can close your terminal.")
print("Monitor it in Vertex AI and Weights & Biases.")
