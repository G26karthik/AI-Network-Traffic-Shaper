from __future__ import annotations

import json
from typing import Dict

import numpy as np
import torch

from .data import FeatureStats, encode_row
from .models import MLP, GRUClassifier


class TorchPredictor:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        obj = torch.load(model_path, map_location=self.device)
        self.model_type = obj["model_type"]
        in_dim = int(obj["in_dim"])
        n_classes = int(obj["n_classes"])
        self.class_to_idx: Dict[str, int] = obj["class_to_idx"]
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.feature_stats = FeatureStats.from_json(obj["feature_stats"]) if isinstance(obj["feature_stats"], str) else FeatureStats.from_json(json.dumps(obj["feature_stats"]))

        if self.model_type == "mlp":
            self.model = MLP(in_dim=in_dim, n_classes=n_classes)
            self._expects_seq = False
        else:
            self.model = GRUClassifier(in_dim=in_dim, n_classes=n_classes)
            self._expects_seq = True
        self.model.load_state_dict(obj["state_dict"])  # type: ignore[arg-type]
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_label(self, features: Dict[str, object]) -> str:
        x = encode_row(features, self.feature_stats.protocol_vocab)
        x = (x - self.feature_stats.mean) / self.feature_stats.std
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        if self._expects_seq:
            xt = xt.unsqueeze(0).unsqueeze(1)  # (1, T=1, F)
        else:
            xt = xt.unsqueeze(0)  # (1, F)
        logits = self.model(xt)
        pred_idx = int(logits.argmax(dim=-1).item())
        return self.idx_to_class[pred_idx]
