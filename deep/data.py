from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# FIXED: Removed dst_port to prevent label leakage
FeatureNames = ["protocol", "length", "src_port"]


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray
    protocol_vocab: Dict[str, int]

    def to_json(self) -> str:
        return json.dumps({
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "protocol_vocab": self.protocol_vocab,
        })

    @staticmethod
    def from_json(s: str) -> "FeatureStats":
        obj = json.loads(s)
        return FeatureStats(
            mean=np.asarray(obj["mean"], dtype=np.float32),
            std=np.asarray(obj["std"], dtype=np.float32),
            protocol_vocab={k: int(v) for k, v in obj["protocol_vocab"].items()},
        )


def build_protocol_vocab(series: pd.Series) -> Dict[str, int]:
    uniq = sorted({str(x) for x in series.dropna().astype(str).unique().tolist()})
    return {name: i for i, name in enumerate(uniq)}


def encode_row(row: pd.Series, proto_vocab: Dict[str, int]) -> np.ndarray:
    proto = str(row.get("protocol", "UNK"))
    proto_idx = proto_vocab.get(proto, -1)
    # protocol index as numeric feature (simple, avoids sparse one-hot)
    # FIXED: Removed dst_port to prevent label leakage
    feats = np.array([
        float(proto_idx if proto_idx >= 0 else -1),
        float(row.get("length", 0.0)),
        float(row.get("src_port", 0.0)),
    ], dtype=np.float32)
    return feats


class PacketDataset(Dataset):
    """Simple per-packet dataset for MLP training.

    Expects a DataFrame with FeatureNames and a 'label' column.
    """

    def __init__(self, df: pd.DataFrame, class_to_idx: Dict[str, int] | None = None, proto_vocab: Dict[str, int] | None = None, feature_stats: FeatureStats | None = None):
        self.df = df.reset_index(drop=True).copy()
        if class_to_idx is None:
            classes = sorted(self.df["label"].astype(str).unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.proto_vocab = proto_vocab or build_protocol_vocab(self.df["protocol"])
        X = np.stack([encode_row(self.df.iloc[i], self.proto_vocab) for i in range(len(self.df))])
        y = self.df["label"].astype(str).map(self.class_to_idx).astype(int).to_numpy()

        if feature_stats is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-6
            self.feature_stats = FeatureStats(mean=mean, std=std, protocol_vocab=self.proto_vocab)
        else:
            self.feature_stats = feature_stats

        self.X = ((X - self.feature_stats.mean) / self.feature_stats.std).astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


def make_loaders(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, batch_size: int = 128) -> Tuple[PacketDataset, PacketDataset, DataLoader, DataLoader]:
    from sklearn.model_selection import train_test_split

    # stratify if possible
    y = df["label"].astype(str)
    vc = y.value_counts()
    stratify = y if (y.nunique() > 1 and (vc.min() >= 2)) else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify)

    train_ds = PacketDataset(train_df)
    test_ds = PacketDataset(test_df, class_to_idx=train_ds.class_to_idx, proto_vocab=train_ds.proto_vocab, feature_stats=train_ds.feature_stats)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_ds, test_ds, train_dl, test_dl

