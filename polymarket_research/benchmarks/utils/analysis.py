"""Lightweight analysis helpers shared by benchmark builders."""

from __future__ import annotations

import pandas as pd


def confidence_slice(probability: float | int | None) -> str:
    """Bucket confidence by distance from 0.5 for lightweight analysis views."""
    if probability is None or pd.isna(probability):
        return "unknown"

    p = float(probability)
    confidence = max(p, 1.0 - p)
    if confidence < 0.60:
        return "50-60"
    if confidence < 0.75:
        return "60-75"
    if confidence < 0.90:
        return "75-90"
    return "90-100"
