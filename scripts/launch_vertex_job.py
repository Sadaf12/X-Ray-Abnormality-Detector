from google.cloud import aiplatform

PROJECT_ID = "project-a5834adc-8f24-44e6-8f1"
REGION = "us-central1"
IMAGE_URI = f"us-central1-docker.pkg.dev/{PROJECT_ID}/xray-abnormality-detector/xray-train:latest"
BUCKET = "mlops_xray_zh"

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.CustomContainerTrainingJob(
    display_name="xray-abnormality-detector-training",
    container_uri=IMAGE_URI,
)

job.run(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    environment_variables={
        "WANDB_API_KEY": "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",   # replace this
        "GOOGLE_APPLICATION_CREDENTIALS": "/app/key.json",
        "GCS_BUCKET": BUCKET,
    },
    replica_count=1,
    sync=True,
)

print("Training job complete.")