import io
import os
import json
import torch
import wandb
import optuna
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from google.cloud import storage

from src.data.dataset import ChestXrayDataset, make_splits
from src.data.transforms import get_train_transform, get_val_transform
from src.models.model import XRayClassifier
from src.training.evaluate import evaluate


def pull_csv_from_gcs(bucket_name, local_path):
    gcs_path = "data/processed/binary_labels.csv"
    print(f"Downloading gs://{bucket_name}/{gcs_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print("CSV downloaded.")

def download_images_from_gcs(data_bucket):
    """Copy all images from GCS to local disk once at job startup."""
    print("Copying images from GCS to local disk...")
    import subprocess
    result = subprocess.run([
        "gsutil", "-m", "cp", "-r",
        f"gs://{data_bucket}/data/raw",
        "/app/data/"
    ], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Error:", result.stderr)
        raise RuntimeError("Image download failed")
    print("Images ready at /app/data/raw/")

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


def build_model(model_name, dropout, lr, weight_decay, pos_weight, device):
    model     = XRayClassifier(model_name, dropout=dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return model, criterion, optimizer


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


def run_hparam_search(cfg, data_bucket, n_trials):
    """Bayesian hyperparameter search using Optuna."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, _ = make_splits(
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

    # Check gsutil is available
    import subprocess
    result = subprocess.run(["which", "gsutil"], capture_output=True, text=True)
    print(f"gsutil location: {result.stdout.strip()}")
    if not result.stdout.strip():
        print("ERROR: gsutil not found!")

    print("Starting CSV download...")
    pull_csv_from_gcs(data_bucket, "/app/data/processed/binary_labels.csv")
    print("CSV download complete.")

    print("Starting image download...")
    download_images_from_gcs(data_bucket)
    print("Image download complete.")

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

    train_df, val_df, _ = make_splits(
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

    wandb.finish()


if __name__ == "__main__":
    main()