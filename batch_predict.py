"""
batch_predict.py

Load a saved sklearn Pipeline (traffic_model.pkl) and run predictions on an existing CSV
(e.g., traffic_features.csv). Optionally print a classification report if the CSV includes labels.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch predict/evaluate using a saved model pipeline")
    p.add_argument("--model", default="traffic_model.pkl", help="Path to model pipeline")
    p.add_argument("--data", default="traffic_features.csv", help="CSV with features (and optionally label)")
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        print(f"[x] Model not found: {args.model}")
        return 2
    if not os.path.exists(args.data):
        print(f"[x] Data not found: {args.data}")
        return 2

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.data)

    need_cols = {"protocol", "length", "src_port", "dst_port"}
    if not need_cols.issubset(df.columns):
        print(f"[x] CSV missing required columns: {need_cols - set(df.columns)}")
        return 2

    X = df[["protocol", "length", "src_port", "dst_port"]].copy()
    y = df["label"].astype(str) if "label" in df.columns else None

    y_pred = pipe.predict(X)
    print(f"[+] Predicted {len(y_pred)} rows. Sample: {y_pred[:10]}")

    if y is not None:
        print("\nClassification Report (full file):")
        print(classification_report(y, y_pred, zero_division=0))
        print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
