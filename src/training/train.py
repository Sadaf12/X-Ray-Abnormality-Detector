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


def pull_data_from_gcs(bucket_name, prefix, local_base="/app"):
    """Pull data from GCS to local container."""
    print(f"Pulling data from gs://{bucket_name}/{prefix}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    if not blobs:
        print(f"  WARNING: no files found at gs://{bucket_name}/{prefix}")
        return

    for blob in blobs:
        local_path = os.path.join(local_base, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"  Downloaded: {local_path}")
    
    print("Data pull complete.")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Pull data from GCS into /app directory
    bucket = os.environ.get("GCS_BUCKET", "mlops_xray_zh")
    pull_data_from_gcs(bucket, prefix="data/processed/binary_labels.csv", local_base="/app")
    pull_data_from_gcs(bucket, prefix="data/raw/nih_chest_xray/", local_base="/app")

    # Change to /app so all relative paths work
    os.chdir("/app")

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=dict(cfg),
        name=f"{cfg.model.name}_bs{cfg.training.batch_size}_lr{cfg.optimizer.lr}",
    )

    # Data
    train_df, val_df, _ = make_splits(
        cfg.data.csv_path,
        cfg.data.val_size,
        cfg.data.test_size,
        cfg.data.random_state,
    )
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(
        ChestXrayDataset(train_df, transform=get_train_transform()),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )
    val_loader = DataLoader(
        ChestXrayDataset(val_df, transform=get_val_transform()),
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )

    # Model
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

    # Training loop
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

    # Push model to GCS
    print("Uploading model to GCS...")
    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(f"models/{cfg.paths.checkpoint_name}")
    blob.upload_from_filename(checkpoint_path)
    print(f"Model saved to gs://{bucket}/models/{cfg.paths.checkpoint_name}")

    wandb.finish()


if __name__ == "__main__":
    main()