"""Bootstrap confidence intervals for binary ROC AUC."""

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    labels: Sequence[int],
    probabilities: Sequence[float],
    repetitions: int = 1000,
    fraction: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return percentile 95% AUC bounds from bootstrap samples."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    sample_size = max(2, round(len(labels) * fraction))
    scores = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=sample_size)
        if np.unique(labels[indices]).size == 2:
            scores.append(roc_auc_score(labels[indices], probabilities[indices]))
    if not scores:
        return float("nan"), float("nan")
    low, high = np.percentile(scores, [2.5, 97.5]).tolist()
    return float(low), float(high)
