import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# Load from .env file
load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = "europe-west4"
IMAGE_URI = f"europe-west4-docker.pkg.dev/{PROJECT_ID}/xray-abnormality-detector-eu/xray-train:latest"
BUCKET = os.environ["GCS_BUCKET"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET}",
)

job = aiplatform.CustomContainerTrainingJob(
    display_name="xray-smoke-test",
    container_uri=IMAGE_URI,
)

job.run(
    machine_type="n1-standard-4",
    environment_variables={
        "WANDB_API_KEY": WANDB_API_KEY,
        "GCS_BUCKET": BUCKET,
        "SMOKE_TEST": "true",
    },
    replica_count=1,
    sync=True,
)

print("Smoke test complete.")