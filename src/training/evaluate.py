import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    all_probs, all_labels, losses = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    return {
        "loss":        float(np.mean(losses)),
        "auc":         float(roc_auc_score(all_labels, all_probs)),
        "f1":          float(f1_score(all_labels, preds)),
        "sensitivity": float(tp / (tp + fn)),  # recall for abnormal — most critical
        "specificity": float(tn / (tn + fp)),
    }