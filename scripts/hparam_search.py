import os
import torch
import optuna
import wandb
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID = os.environ["PROJECT_ID"]
REGION = "europe-west4"
IMAGE_URI = f"europe-west4-docker.pkg.dev/{PROJECT_ID}/xray-abnormality-detector-eu/xray-train:latest"
DATA_BUCKET = os.environ["DATA_BUCKET"]
STAGING_BUCKET = os.environ["STAGING_BUCKET"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{STAGING_BUCKET}",
)

def launch_trial(trial):
    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    model_name   = trial.suggest_categorical("model_name",
                       ["efficientnet_b0", "efficientnet_b2"])

    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"xray-hparam-trial-{trial.number}",
        container_uri=IMAGE_URI,
    )

    job.run(
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        environment_variables={
            "WANDB_API_KEY":  WANDB_API_KEY,
            "DATA_BUCKET":    DATA_BUCKET,
            "SMOKE_TEST":     "false",
            "HPARAM_TRIAL":   "true",        # 5 epochs, 20% data
            "LR":             str(lr),
            "DROPOUT":        str(dropout),
            "WEIGHT_DECAY":   str(weight_decay),
            "BATCH_SIZE":     str(batch_size),
            "MODEL_NAME":     model_name,
            "TRIAL_NUMBER":   str(trial.number),
        },
        replica_count=1,
        sync=True,   # wait for result before next trial
    )

    # Get AUC from W&B
    api = wandb.Api()
    runs = api.runs(
        "sadafambreen-zhaw/X-Ray-Abnormality-Detector",
        filters={"display_name": f"trial-{trial.number}"}
    )
    best_auc = max(r.summary.get("val/auc", 0) for r in runs) if runs else 0
    return best_auc

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    study_name="xray-hparam-v1",
    storage="sqlite:///optuna_study.db",
    load_if_exists=True,
)
study.optimize(launch_trial, n_trials=10)

print("\nBest params:", study.best_params)
print(f"Best AUC: {study.best_value:.4f}")