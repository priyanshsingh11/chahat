"""
SF-IDS MVP - Step 2: Training + evaluation (run this on Google Colab GPU)
Reads artifacts/dataset.pt, trains DCNN-BiLSTM, evaluates on the held-out test set,
and writes the model + metrics + confusion matrix into artifacts/.

Usage:  python train.py            (optional: python train.py --epochs 15)
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from model import DCNNBiLSTM

ART = "artifacts"


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(1).cpu())
            trues.append(yb)
    return torch.cat(trues).numpy(), torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-cap", type=float, default=10.0,
                    help="Max class weight clamp (lower = fewer false alarms)")
    args = ap.parse_args()

    torch.manual_seed(42)
    data = torch.load(f"{ART}/dataset.pt")
    meta = json.load(open(f"{ART}/meta.json"))
    classes = meta["classes"]
    n_feat, n_cls = data["X_train"].shape[1], len(classes)

    if torch.cuda.is_available():
        device = torch.device("cuda")          # Colab / NVIDIA
    elif torch.backends.mps.is_available():
        device = torch.device("mps")           # Apple Silicon GPU (M1-M5)
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | features={n_feat} classes={n_cls}")

    train_loader = DataLoader(TensorDataset(data["X_train"], data["y_train"]),
                              batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(data["X_test"], data["y_test"]),
                             batch_size=2048)

    # Class-weighted loss handles imbalance (gentler clamp cuts false alarms)
    counts = torch.bincount(data["y_train"], minlength=n_cls).float()
    weights = (counts.sum() / (n_cls * counts.clamp(min=1))).clamp(max=args.weight_cap).to(device)

    model = DCNNBiLSTM(n_feat, n_cls).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        scheduler.step()
        y_true, y_pred = evaluate(model, test_loader, device)
        acc = accuracy_score(y_true, y_pred)
        avg = total / len(train_loader.dataset)
        history.append({"epoch": epoch, "loss": avg, "test_acc": acc})
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {avg:.4f} | test acc {acc:.4f}")
    train_secs = time.time() - t0

    # --- Final evaluation ---------------------------------------------------
    y_true, y_pred = evaluate(model, test_loader, device)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=range(n_cls),
                                   target_names=classes, zero_division=0, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=range(n_cls))
    print("\n" + report)

    # Inference latency (CPU, single flow) - what a serverless node would see
    cpu_model = DCNNBiLSTM(n_feat, n_cls)
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()
    one = data["X_test"][:1]
    with torch.no_grad():
        for _ in range(10):
            cpu_model(one)
        t = time.time()
        for _ in range(200):
            cpu_model(one)
    latency_ms = (time.time() - t) / 200 * 1000

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p, "macro_recall": r, "macro_f1": f1,
        "train_seconds": train_secs, "cpu_latency_ms_per_flow": latency_ms,
        "epochs": args.epochs, "classes": classes, "history": history,
        "confusion_matrix": cm.tolist(),
    }
    json.dump(metrics, open(f"{ART}/metrics.json", "w"), indent=2)
    open(f"{ART}/classification_report.txt", "w").write(report)

    torch.save({"state_dict": cpu_model.state_dict(),
                "num_features": n_feat, "num_classes": n_cls}, f"{ART}/sf_ids_model.pth")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(1.2 * n_cls + 3, 1.0 * n_cls + 2.5))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n_cls), classes, rotation=45, ha="right")
        ax.set_yticks(range(n_cls), classes)
        for i in range(n_cls):
            for j in range(n_cls):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("SF-IDS DCNN-BiLSTM - confusion matrix (test set)")
        fig.tight_layout()
        fig.savefig(f"{ART}/confusion_matrix.png", dpi=150)
    except Exception as e:
        print("Skipped plot:", e)

    print(f"\nAccuracy {metrics['accuracy']:.4f} | macro F1 {f1:.4f} | "
          f"CPU latency {latency_ms:.2f} ms/flow | trained in {train_secs:.0f}s")
    print(f"Saved {ART}/sf_ids_model.pth, metrics.json, classification_report.txt, confusion_matrix.png")


if __name__ == "__main__":
    main()
