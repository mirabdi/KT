"""Streaming evaluation metrics for binary response prediction."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


class MetricAccumulator:
    """Collects masked predictions on the CPU (no device memory is retained)."""

    def __init__(self) -> None:
        self._probs: List[np.ndarray] = []
        self._targets: List[np.ndarray] = []
        self.loss_sum = 0.0
        self.tokens = 0
        self._has_loss = False

    def update(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
               loss_sum: float | None = None) -> None:
        m = mask.bool()
        probs = torch.sigmoid(logits.detach().float())[m].cpu().numpy()
        tgt = targets[m].detach().cpu().numpy().astype(np.float32)
        self._probs.append(probs)
        self._targets.append(tgt)
        self.tokens += int(len(tgt))
        if loss_sum is not None:
            self.loss_sum += float(loss_sum)
            self._has_loss = True

    def compute(self) -> Dict[str, float]:
        if not self._probs:
            return {}
        p = np.concatenate(self._probs)
        y = np.concatenate(self._targets)
        out = compute_metrics(y, p)
        if self._has_loss:
            out["loss"] = self.loss_sum / max(self.tokens, 1)
        return out


def compute_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """AUC, RMSE, accuracy, MAE and binary log-loss."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    y = y.astype(np.float64)
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else math.nan
    return {
        "auc": float(auc),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "acc": float(accuracy_score(y, p > 0.5)),
        "mae": float(np.mean(np.abs(y - p))),
        "bce": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "n": int(len(y)),
    }
