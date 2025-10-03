from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or [64, 64]
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_size: int = 64, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=0.0 if num_layers == 1 else dropout)
        feat = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)
