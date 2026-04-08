"""Small metric helpers used in notebooks and experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def clipped_probabilities(values, *, eps: float = 1e-6) -> np.ndarray:
    """Clip probabilities away from 0 and 1 for stable metrics."""
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def safe_auc(y_true, score) -> float:
    """Return ROC-AUC when both classes are present, else NaN."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, score)


def safe_average_precision(y_true, score) -> float:
    """Return average precision when both classes are present, else NaN."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, score)
