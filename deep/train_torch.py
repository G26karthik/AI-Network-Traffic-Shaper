from __future__ import annotations

"""
Train deep learning models (MLP/GRU) on the project's CSV dataset.

Usage (PowerShell):
  ./traffic_env/Scripts/python.exe ./deep/train_torch.py --data dataset.csv --model-out deep_model.pt

Reports: accuracy, precision, recall, F1, confusion matrix.
Saves: model .pt, label mapping, and feature stats JSON next to the model.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .data import FeatureNames, PacketDataset, make_loaders
from .models import MLP, GRUClassifier


def load_dataframe(path: str, filter_other: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Coerce numeric
    for c in ["length", "src_port", "dst_port"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FeatureNames + ["label"]).copy()
    if filter_other:
        df = df[df["label"].isin(["VoIP", "FTP", "HTTP"])].copy()
    if df.empty:
        raise ValueError("No rows after cleaning/filtering")
    return df


def train_epoch(model: torch.nn.Module, dl: DataLoader, device: torch.device, optimizer, criterion) -> float:
    model.train()
    total = 0.0
    n = 0
    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb) if xb.ndim == 2 else model(xb)  # supports (B,F) or (B,T,F)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * yb.size(0)
        n += yb.size(0)
    return total / max(1, n)


@torch.no_grad()
def evaluate(model: torch.nn.Module, dl: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb) if xb.ndim == 2 else model(xb)
        pred = logits.argmax(dim=-1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train PyTorch model on traffic dataset")
    p.add_argument("--data", default="dataset.csv", help="CSV dataset path")
    p.add_argument("--model-out", default="deep_model.pt", help="Output .pt path")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--filter-other", action="store_true", help="Filter out 'Other' label")
    p.add_argument("--class-weight", action="store_true", help="Use class-weighted loss")
    args = p.parse_args(argv)

    df = load_dataframe(args.data, filter_other=args.filter_other)
    train_ds, test_ds, train_dl, test_dl = make_loaders(df, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIXED: Reduced from 4 to 3 features (removed dst_port)
    in_dim = 3  # protocol_idx, length, src_port
    n_classes = len(train_ds.class_to_idx)
    if args.model == "mlp":
        model = MLP(in_dim=in_dim, n_classes=n_classes)
        input_is_sequence = False
    else:
        model = GRUClassifier(in_dim=in_dim, n_classes=n_classes)
        input_is_sequence = True
    model.to(device)

    if args.class_weight:
        # Compute weights inversely proportional to class freq
        counts = np.bincount(train_ds.y, minlength=n_classes).astype(np.float32)
        weights = (counts.sum() / np.maximum(counts, 1.0))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_dl, device, optimizer, criterion)
        y_true, y_pred = evaluate(model, test_dl, device)
        acc = accuracy_score(y_true, y_pred)
        p_, r_, f1_, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | acc={acc:.3f} | P/R/F1={p_:.3f}/{r_:.3f}/{f1_:.3f}")

    # Final report
    y_true, y_pred = evaluate(model, test_dl, device)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(n_classes)], zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "n_classes": n_classes,
        "model_type": args.model,
        "class_to_idx": train_ds.class_to_idx,
        "feature_stats": train_ds.feature_stats.to_json(),
    }, args.model_out)
    meta_path = os.path.splitext(args.model_out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "idx_to_class": {int(k): v for k, v in {v: k for k, v in train_ds.class_to_idx.items()}.items()},
            "feature_names": FeatureNames,
        }, f, indent=2)
    print(f"Saved model -> {args.model_out} and meta -> {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
