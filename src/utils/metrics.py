"""
Evaluation metrics for multimodal learning.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from typing import List, Union


def compute_accuracy(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted labels or logits (B,) or (B, C)
        targets: Ground truth labels (B,)

    Returns:
        Accuracy as a float in [0, 1]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # If predictions are logits, convert to labels
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)

    return accuracy_score(targets, predictions)


def compute_f1(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    average: str = "macro",
) -> float:
    """
    Compute F1 score.

    Args:
        predictions: Predicted labels or logits (B,) or (B, C)
        targets: Ground truth labels (B,)
        average: Averaging strategy ('macro', 'micro', 'weighted')

    Returns:
        F1 score as a float in [0, 1]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # If predictions are logits, convert to labels
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)

    return f1_score(targets, predictions, average=average)


class MetricTracker:
    """Track multiple metrics across batches."""

    def __init__(self, metrics: List[str] = None):
        """
        Args:
            metrics: List of metric names to track
        """
        if metrics is None:
            metrics = ["accuracy", "f1_macro"]

        self.metrics = metrics
        self.predictions = []
        self.targets = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add batch predictions and targets."""
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> dict:
        """Compute all tracked metrics."""
        if not self.predictions:
            return {m: 0.0 for m in self.metrics}

        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)

        results = {}
        for metric in self.metrics:
            if metric == "accuracy":
                results[metric] = compute_accuracy(all_preds, all_targets)
            elif metric == "f1_macro":
                results[metric] = compute_f1(all_preds, all_targets, average="macro")
            elif metric == "f1_micro":
                results[metric] = compute_f1(all_preds, all_targets, average="micro")
            elif metric == "f1_weighted":
                results[metric] = compute_f1(all_preds, all_targets, average="weighted")

        return results

    def reset(self) -> None:
        """Reset tracked data."""
        self.predictions = []
        self.targets = []
