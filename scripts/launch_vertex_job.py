import os
from dotenv import load_dotenv
from google.cloud import aiplatform

# Load from .env file
load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = "us-central1"
IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/xray-abnormality-detector/xray-train:latest"
BUCKET = os.environ["GCS_BUCKET"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

aiplatform.init(project=PROJECT_ID, location=REGION)

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