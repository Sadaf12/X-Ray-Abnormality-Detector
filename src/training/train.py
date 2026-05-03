import io
import os
import torch
import wandb
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from google.cloud import storage

from src.data.dataset import ChestXrayDataset, make_splits
from src.data.transforms import get_train_transform, get_val_transform
from src.models.model import XRayClassifier
from src.training.evaluate import evaluate


def pull_csv_from_gcs(bucket_name, local_path):
    """Download CSV directly from GCS."""
    gcs_path = "data/processed/binary_labels.csv"
    print(f"Downloading gs://{bucket_name}/{gcs_path} to {local_path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print("CSV downloaded successfully.")


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


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    os.chdir("/app")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bucket = os.environ.get("GCS_BUCKET", "mlops_xray_zh")

    # Download only the CSV (9MB) — images stream directly from GCS
    pull_csv_from_gcs(bucket, "/app/data/processed/binary_labels.csv")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=dict(cfg),
        name=f"{cfg.model.name}_bs{cfg.training.batch_size}_lr{cfg.optimizer.lr}",
    )

    train_df, val_df, _ = make_splits(
        cfg.data.csv_path,
        cfg.data.val_size,
        cfg.data.test_size,
        cfg.data.random_state,
    )

    # Smoke test mode — 500 images, 2 epochs
    if os.environ.get("SMOKE_TEST") == "true":
        print("SMOKE TEST MODE — 500 images, 2 epochs")
        train_df = train_df.sample(500, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(100, random_state=42).reset_index(drop=True)
        cfg.training.epochs = 2

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        ChestXrayDataset(train_df, transform=get_train_transform(),
                         gcs_bucket=bucket),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        ChestXrayDataset(val_df, transform=get_val_transform(),
                         gcs_bucket=bucket),
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model = XRayClassifier(
        model_name=cfg.model.name,
        dropout=cfg.model.dropout,
        pretrained=cfg.model.pretrained,
    ).to(device)

    pos_weight = torch.tensor([cfg.loss.pos_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.scheduler.T_max
    )

    os.makedirs(cfg.paths.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.paths.model_dir, cfg.paths.checkpoint_name)
    best_auc = 0.0

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"sensitivity={val_metrics['sensitivity']:.4f}"
        )

        wandb.log({
            "epoch":           epoch + 1,
            "train/loss":      train_loss,
            "val/loss":        val_metrics["loss"],
            "val/auc":         val_metrics["auc"],
            "val/f1":          val_metrics["f1"],
            "val/sensitivity": val_metrics["sensitivity"],
            "val/specificity": val_metrics["specificity"],
            "lr":              scheduler.get_last_lr()[0],
        })

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best AUC {best_auc:.4f} — checkpoint saved")

    # Upload model to GCS
    print("Uploading model checkpoint to GCS...")
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(f"models/{cfg.paths.checkpoint_name}")
    blob.upload_from_filename(checkpoint_path)
    print(f"Model saved to gs://{bucket}/models/{cfg.paths.checkpoint_name}")

    wandb.finish()


if __name__ == "__main__":
    main()