"""Training entrypoint with MLflow experiment tracking."""

import argparse
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.dataset import ECGBeatDataset
from src.models.cnn import ECGCNN, count_parameters


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_class_weights(counts):
    weights = counts.sum() / (len(counts) * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * X.size(0)
        total += X.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    accuracy = (preds == labels).mean()
    return total_loss / total, accuracy, macro_f1, preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cnn_baseline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = ECGBeatDataset("data/processed/train.npz")
    val_ds = ECGBeatDataset("data/processed/val.npz")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = ECGCNN(num_classes=5, dropout=cfg["dropout"]).to(device)

    # Square-root scaling on class weights reduces minority-class over-prediction
    raw_weights = compute_class_weights(train_ds.class_counts())
    class_weights = torch.sqrt(raw_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    mlflow.set_experiment(cfg["experiment_name"])
    with mlflow.start_run(run_name=cfg.get("run_name")):
        mlflow.log_params(cfg)
        mlflow.log_param("n_params", count_parameters(model))
        mlflow.log_param("device", str(device))
        mlflow.log_param("class_weights", class_weights.tolist())

        best_f1 = 0.0
        best_state = None

        for epoch in range(cfg["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(f"Epoch {epoch+1:2d}/{cfg['epochs']} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_ds = ECGBeatDataset("data/processed/test.npz")
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
        test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, criterion, device)

        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_macro_f1", test_f1)
        mlflow.log_metric("best_val_macro_f1", best_f1)

        report = classification_report(labels, preds, target_names=["N", "S", "V", "F", "Q"], zero_division=0)
        cm = confusion_matrix(labels, preds)
        print("\nTest set classification report:\n", report)
        print("Confusion matrix:\n", cm)

        Path("artifacts").mkdir(exist_ok=True)
        with open("artifacts/test_report.txt", "w") as f:
            f.write(report + "\n\nConfusion matrix:\n" + str(cm))
        mlflow.log_artifact("artifacts/test_report.txt")

        mlflow.pytorch.log_model(model, artifact_path="model")

        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/best.pt")

        print(f"\n✅ Run complete. Best val F1: {best_f1:.4f}. Test F1: {test_f1:.4f}.")


if __name__ == "__main__":
    main()
