"""Shared helpers for release-facing reference baselines."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from polymarket_research.benchmarks.utils.splits import select_split_rows


def require_targets(
    targets: pd.DataFrame,
    *,
    split: str,
    baseline_name: str,
) -> pd.DataFrame:
    """Return a clean target frame or raise when the requested split is empty."""
    frame = targets.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError(f"{baseline_name} requires non-empty targets for split={split!r}.")
    return frame


def benchmark_targets_frame(
    benchmark,
    *,
    split: str,
    baseline_name: str,
) -> pd.DataFrame:
    """Return a clean split-specific target frame from a frozen benchmark object."""
    return require_targets(
        select_split_rows(benchmark.targets_frame, split),
        split=split,
        baseline_name=baseline_name,
    )


def numeric_mode(series: pd.Series) -> int:
    """Return the first modal integer value from a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError("Cannot compute a majority label from an empty numeric series.")
    return int(values.mode().iloc[0])


def dataclass_manifest(value: Any, *, name: str, train_split: str) -> dict[str, Any]:
    """Serialize a fitted baseline into a compact JSON-like manifest."""
    return {
        "name": name,
        "train_split": str(train_split),
        "parameters": asdict(value),
    }
