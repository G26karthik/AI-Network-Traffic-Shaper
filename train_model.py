"""
train_model.py

Train a classifier on features in dataset.csv produced by capture_features.py.
Features used: protocol (categorical), length, src_port, dst_port (numeric)
Labels: strings (e.g., VoIP, FTP, HTTP). Optionally filter out 'Other'.

Outputs: a single sklearn Pipeline saved to traffic_model.pkl.
"""

from __future__ import annotations

import argparse
import os
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_prepare(data_path: str, filter_other: bool = True) -> tuple[pd.DataFrame, pd.Series]:
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Dataset not found: {data_path}. Run capture_features.py first.")
	df = pd.read_csv(data_path)
	# Basic cleaning / type coercion
	for col in ["length", "src_port", "dst_port"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=["length", "src_port", "dst_port", "protocol", "label"]).copy()

	if filter_other:
		df = df[df["label"].isin(["VoIP", "FTP", "HTTP"])].copy()
	if df.empty:
		raise ValueError("No rows available after filtering/cleaning. Collect more data.")

	X = df[["protocol", "length", "src_port", "dst_port"]]
	y = df["label"].astype(str)
	return X, y


def build_pipeline() -> Pipeline:
	numeric_features = ["length", "src_port", "dst_port"]
	categorical_features = ["protocol"]

	pre = ColumnTransformer(
		transformers=[
			("num", StandardScaler(), numeric_features),
			("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
		]
	)

	clf = RandomForestClassifier(n_estimators=200, random_state=42)

	pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
	return pipe


def main(argv: List[str] | None = None) -> int:
	p = argparse.ArgumentParser(description="Train traffic classifier on dataset.csv")
	p.add_argument("--data", default="dataset.csv", help="Path to dataset CSV (default dataset.csv)")
	p.add_argument("--model-out", default="traffic_model.pkl", help="Output model path (default traffic_model.pkl)")
	p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
	p.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
	p.add_argument("--keep-other", action="store_true", help="Keep 'Other' labeled rows in training")
	args = p.parse_args(argv)

	try:
		X, y = load_and_prepare(args.data, filter_other=not args.keep_other)
	except Exception as e:
		print(f"[x] Failed to load/prepare dataset: {e}")
		return 2

	# Use stratify only if all classes have at least 2 samples
	vc = y.value_counts()
	stratify = y if (y.nunique() > 1 and (vc.min() >= 2)) else None
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
	)

	pipe = build_pipeline()
	pipe.fit(X_train, y_train)

	y_pred = pipe.predict(X_test)
	# Fix label ordering for metrics
	classes = ["VoIP", "FTP", "HTTP"] if set(["VoIP", "FTP", "HTTP"]).issuperset(set(y.unique())) else sorted(y.unique())

	print("✅ Model Training Completed")
	print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=classes))
	print(
		"\nClassification Report:\n",
		classification_report(y_test, y_pred, labels=classes, target_names=classes, zero_division=0),
	)

	joblib.dump(pipe, args.model_out)
	print(f"\n✅ Saved model pipeline to {args.model_out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
