import io
import os
import json
import torch
import wandb
import optuna
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from google.cloud import storage

from src.data.dataset import ChestXrayDataset, make_splits
from src.data.transforms import get_train_transform, get_val_transform
from src.models.model import XRayClassifier
from src.training.evaluate import evaluate

# downloads the binary_labels.csv file from a Google Cloud Storage (GCS) bucket to the local machine, 
# but only if the file does not already exist locally.

def pull_csv_from_gcs(bucket_name, local_path):
    """Download CSV from GCS — skips if already exists."""
    if os.path.exists(local_path):
        print(f"CSV already exists at {local_path} — skipping download.")
        return

    gcs_path = "data/processed/binary_labels.csv"
    print(f"Downloading gs://{bucket_name}/{gcs_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print("CSV downloaded.")

# copies the chest X-ray image dataset from Google Cloud Storage to the local disk using gsutil.
# checks whether the images are already available locally to avoid unnecessary copying.

def download_images_from_gcs(data_bucket, local_base=None):
    """Copy images from GCS to local disk — skips if already downloaded."""
    local_check = "/app/data/raw/nih_chest_xray"
    
    if os.path.exists(local_check) and len(os.listdir(local_check)) > 0:
        print("Images already on local disk — skipping download.")
        return
    
    print("Copying images from GCS to local disk (parallel)...")
    import subprocess
    result = subprocess.run([
        "gsutil", "-m", "cp", "-r",
        f"gs://{data_bucket}/data/raw",
        "/app/data/"
    ])
    if result.returncode != 0:
        raise RuntimeError("Image download failed")
    print("Images ready at /app/data/raw/")

# Tperforms one full training pass over the training dataset. 
# It loads batches of images and labels, moves them to the GPU or CPU, computes predictions and loss, 
# performs backpropagation, updates the model weights using the optimizer, 
# and returns the average training loss for the epoch.

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# creates PyTorch DataLoaders for the training and validation datasets. 
# It applies training transforms (including augmentation) to the training data and validation transforms to the validation data, 
# then prepares the data in batches for efficient loading during training and evaluation.

def build_loaders(train_df, val_df, batch_size, data_bucket):
    train_loader = DataLoader(
        ChestXrayDataset(train_df, transform=get_train_transform(),
                         data_bucket=data_bucket),
        batch_size=batch_size, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        ChestXrayDataset(val_df, transform=get_val_transform(),
                         data_bucket=data_bucket),
        batch_size=batch_size * 2, shuffle=False, num_workers=2,
    )
    return train_loader, val_loader

# creates the X-ray classification model, defines the loss function, and initializes the optimizer. 
# builds the XRayClassifier with the specified architecture and dropout rate, moves it to the selected device (CPU or GPU), 
# uses BCEWithLogitsLoss for binary classification with class weighting, and configures the AdamW optimizer with the chosen learning rate and weight decay.

def build_model(model_name, dropout, lr, weight_decay, pos_weight, device):
    model     = XRayClassifier(model_name, dropout=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return model, criterion, optimizer

# Does the same as in notebooks/evaluate_testset.ipynb just now included in the pipeline

def evaluate_test_set(model, test_df, criterion, cfg, device, threshold=0.5):
    test_loader = DataLoader(
        ChestXrayDataset(test_df, transform=get_val_transform()),
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
    )

    model.eval()
    all_probs, all_labels, losses = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.asarray(all_probs)
    all_labels = np.asarray(all_labels).astype(int)
    all_preds = (all_probs >= threshold).astype(int)

    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    threshold_idx = int(np.argmin(np.abs(roc_thresholds - threshold)))

    metrics = {
        "test_loss": float(np.mean(losses)),
        "test_accuracy": float(accuracy_score(all_labels, all_preds)),
        "test_auc": float(roc_auc_score(all_labels, all_probs)),
        "threshold": float(roc_thresholds[threshold_idx]),
        "threshold_label": threshold,
        "tpr": float(tpr[threshold_idx]),
        "fpr": float(fpr[threshold_idx]),
        "best_test_accuracy": float(accuracy_score(all_labels, all_preds)),
    }
    roc_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": roc_thresholds.tolist(),
    }

    return metrics, roc_data


def save_and_upload_evaluation_artifacts(
    metrics, roc_data, train_df, val_df, test_df, cfg, data_bucket
):
    target_counts = (
        train_df["target"].value_counts()
        .add(val_df["target"].value_counts(), fill_value=0)
        .add(test_df["target"].value_counts(), fill_value=0)
        .sort_index()
    )
    total_size = int(len(train_df) + len(val_df) + len(test_df))
    target_distribution = {
        str(int(target)): float(count / total_size)
        for target, count in target_counts.items()
    }

    metrics_payload = {
        "training_data": int(len(train_df)),
        "validation_data": int(len(val_df)),
        "test_data": int(len(test_df)),
        "total_data": total_size,
        "target_distribution": target_distribution,
        **metrics,
    }

    evaluation_dir = os.path.join(cfg.paths.model_dir, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)
    metrics_path = os.path.join(evaluation_dir, "test_metrics.json")
    roc_path = os.path.join(evaluation_dir, "roc_curve.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    with open(roc_path, "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)

    client = storage.Client()
    bucket = client.bucket(data_bucket)
    bucket.blob("models/evaluation/test_metrics.json").upload_from_filename(metrics_path)
    bucket.blob("models/evaluation/roc_curve.json").upload_from_filename(roc_path)

    print(f"Evaluation metrics saved to gs://{data_bucket}/models/evaluation/test_metrics.json")
    print(f"ROC curve data saved to gs://{data_bucket}/models/evaluation/roc_curve.json")

    return metrics_payload


def run_test_evaluation(cfg, data_bucket, device, checkpoint_path=None):
    checkpoint_path = checkpoint_path or os.path.join(
        cfg.paths.model_dir, cfg.paths.checkpoint_name
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Train first or download the checkpoint before evaluation."
        )

    train_df, val_df, test_df = make_splits(
        cfg.data.csv_path, cfg.data.val_size,
        cfg.data.test_size, cfg.data.random_state,
    )
    model, criterion, _ = build_model(
        cfg.model.name, cfg.model.dropout, cfg.optimizer.lr,
        cfg.optimizer.weight_decay, cfg.loss.pos_weight, device
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print("Evaluating model on held-out test set...")
    test_metrics, roc_data = evaluate_test_set(
        model, test_df, criterion, cfg, device, threshold=0.5
    )
    evaluation_payload = save_and_upload_evaluation_artifacts(
        test_metrics, roc_data, train_df, val_df, test_df, cfg, data_bucket
    )
    print(f"Test evaluation: {json.dumps(evaluation_payload, indent=2)}")

    return evaluation_payload

# manages the full training and validation process across multiple epochs. 
# For each epoch, it trains the model on the training dataset, evaluates it on the validation dataset, 
# updates the learning rate scheduler if used, logs performance metrics such as loss, AUC, F1-score, sensitivity, and specificity to Weights & Biases, 
# and tracks the best validation AUC achieved. During hyperparameter search, it also reports results to Optuna and can stop unpromising trials early using pruning.

def run_epochs(model, train_loader, val_loader, optimizer, criterion,
               scheduler, device, epochs, trial=None):
    """Shared epoch loop used by both training and hparam search."""
    best_auc = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step()

        wandb.log({
            "epoch":           epoch + 1,
            "train/loss":      train_loss,
            "val/loss":        metrics["loss"],
            "val/auc":         metrics["auc"],
            "val/f1":          metrics["f1"],
            "val/sensitivity": metrics["sensitivity"],
            "val/specificity": metrics["specificity"],
        })

        print(f"Epoch {epoch+1}/{epochs} | loss={train_loss:.4f} | "
              f"auc={metrics['auc']:.4f} | sensitivity={metrics['sensitivity']:.4f}")

        best_auc = max(best_auc, metrics["auc"])

        # Optuna pruning — only active during hparam search
        if trial is not None:
            trial.report(metrics["auc"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_auc

# performs hyperparameter optimization using Optuna. 
# first splits the dataset into training and validation sets and uses only 20% of the data and 5 epochs to speed up the search process. 
# saves the best parameters and saves it to gcp

def run_hparam_search(cfg, data_bucket, n_trials):
    """Bayesian hyperparameter search using Optuna."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = make_splits(
        cfg.data.csv_path, cfg.data.val_size,
        cfg.data.test_size, cfg.data.random_state,
    )
    # Use 20% of data per trial for speed
    train_df = train_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    def objective(trial):
        lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout      = trial.suggest_float("dropout", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])

        wandb.init(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=f"trial-{trial.number}", group="hparam-search",
            config={"lr": lr, "dropout": dropout,
                    "weight_decay": weight_decay, "batch_size": batch_size},
            reinit=True,
        )

        train_loader, val_loader = build_loaders(
            train_df, val_df, batch_size, data_bucket
        )
        model, criterion, optimizer = build_model(
            cfg.model.name, dropout, lr, weight_decay, cfg.loss.pos_weight, device
        )

        best_auc = run_epochs(
            model, train_loader, val_loader, optimizer,
            criterion, scheduler=None, device=device,
            epochs=5, trial=trial,
        )

        wandb.finish()
        return best_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
        study_name="xray-hparam-v1",
    )
    study.optimize(objective, n_trials=n_trials)

    best = {"best_params": study.best_params, "best_auc": study.best_value}
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best params to GCS
    client = storage.Client()
    bucket = client.bucket(data_bucket)
    bucket.blob("hparam/best_params.json").upload_from_string(json.dumps(best))
    print(f"Saved to gs://{data_bucket}/hparam/best_params.json")

# central entry point of the training pipeline. loads the training configuration using Hydra, 
# sets up the environment and device (CPU/GPU), downloads the dataset from Google Cloud Storage if needed, 
# and decides whether to run hyperparameter optimization or normal model training based on the RUN_MODE environment variable.

# In training mode, it initializes Weights & Biases logging, creates the train and validation splits, builds the data loaders, model, optimizer, 
# and learning rate scheduler, and then trains the model for multiple epochs while evaluating validation performance after each epoch. 
# The best-performing model checkpoint is saved locally and uploaded to Google Cloud Storage at the end of training.

@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("=" * 50)
    print("Container started successfully")
    print(f"Python version: {os.sys.version}")
    print("=" * 50)
    
    os.chdir("/app")
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_bucket = os.environ.get("DATA_BUCKET", "nih-xray-data")
    run_mode    = os.environ.get("RUN_MODE", "train")

    print(f"Device: {device}")
    print(f"Data bucket: {data_bucket}")
    print(f"Run mode: {run_mode}")

    # Evaluation mode uses an existing local checkpoint and local test data.
    if run_mode == "evaluate":
        checkpoint_path = os.environ.get(
            "MODEL_LOCAL_PATH",
            os.path.join(cfg.paths.model_dir, cfg.paths.checkpoint_name),
        )
        run_test_evaluation(cfg, data_bucket, device, checkpoint_path)
        return

    # Check gsutil is available
    import subprocess
    result = subprocess.run(["which", "gsutil"], capture_output=True, text=True)
    print(f"gsutil location: {result.stdout.strip()}")
    if not result.stdout.strip():
        print("ERROR: gsutil not found!")

    print("Starting CSV download...")
    pull_csv_from_gcs(data_bucket, "/app/data/processed/binary_labels.csv")

    print("Starting image download...")
    download_images_from_gcs(data_bucket, local_base=os.getcwd())

    # ── Hparam search mode ──────────────────────────────────────────────────
    if run_mode == "hparam":
        n_trials = int(os.environ.get("N_TRIALS", "10"))
        run_hparam_search(cfg, data_bucket, n_trials)
        return

    # ── Training mode ───────────────────────────────────────────────────────
    wandb.init(
        project=cfg.wandb.project, entity=cfg.wandb.entity,
        config=dict(cfg),
        name=f"{cfg.model.name}_bs{cfg.training.batch_size}_lr{cfg.optimizer.lr}",
    )

    train_df, val_df, test_df = make_splits(
        cfg.data.csv_path, cfg.data.val_size,
        cfg.data.test_size, cfg.data.random_state,
    )

    if os.environ.get("SMOKE_TEST") == "true":
        print("SMOKE TEST MODE — 500 images, 2 epochs")
        train_df = train_df.sample(500, random_state=42).reset_index(drop=True)
        val_df   = val_df.sample(100, random_state=42).reset_index(drop=True)
        cfg.training.epochs = 2

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader, val_loader = build_loaders(
        train_df, val_df, cfg.training.batch_size, data_bucket
    )
    model, criterion, optimizer = build_model(
        cfg.model.name, cfg.model.dropout, cfg.optimizer.lr,
        cfg.optimizer.weight_decay, cfg.loss.pos_weight, device
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler.T_max
    )

    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.paths.model_dir, cfg.paths.checkpoint_name)
    best_auc = 0.0

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics    = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.training.epochs} | "
              f"loss={train_loss:.4f} | auc={metrics['auc']:.4f} | "
              f"sensitivity={metrics['sensitivity']:.4f}")

        wandb.log({
            "epoch":           epoch + 1,
            "train/loss":      train_loss,
            "val/loss":        metrics["loss"],
            "val/auc":         metrics["auc"],
            "val/f1":          metrics["f1"],
            "val/sensitivity": metrics["sensitivity"],
            "val/specificity": metrics["specificity"],
            "lr":              scheduler.get_last_lr()[0],
        })

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best AUC {best_auc:.4f} — checkpoint saved")

    # Upload model to GCS
    print("Uploading model to GCS...")
    client = storage.Client()
    bucket = client.bucket(data_bucket)
    bucket.blob(f"models/{cfg.paths.checkpoint_name}").upload_from_filename(checkpoint_path)
    print(f"Model saved to gs://{data_bucket}/models/{cfg.paths.checkpoint_name}")

    evaluation_payload = run_test_evaluation(
        cfg, data_bucket, device, checkpoint_path
    )
    test_metrics = {
        "test_loss": evaluation_payload["test_loss"],
        "test_accuracy": evaluation_payload["test_accuracy"],
        "test_auc": evaluation_payload["test_auc"],
        "tpr": evaluation_payload["tpr"],
        "fpr": evaluation_payload["fpr"],
    }
    wandb.log({
        "test/loss": test_metrics["test_loss"],
        "test/accuracy": test_metrics["test_accuracy"],
        "test/auc": test_metrics["test_auc"],
        "test/tpr_at_50pct": test_metrics["tpr"],
        "test/fpr_at_50pct": test_metrics["fpr"],
    })
    print(f"Test evaluation: {json.dumps(evaluation_payload, indent=2)}")

    wandb.finish()


if __name__ == "__main__":
    main()
