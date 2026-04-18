"""Shared helpers for clean derived data representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


@dataclass(frozen=True)
class RepresentationFrame:
    """Hold a named derived dataframe and provide simple persistence helpers."""

    name: str
    frame: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Return a compact shape summary for the representation frame."""
        return pd.DataFrame([{"name": self.name, "rows": len(self.frame), "cols": self.frame.shape[1]}])

    def save(self, path: str | Path) -> Path:
        """Persist the representation as a parquet file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(out_path, index=False)
        return out_path

    @classmethod
    def from_parquet(cls, name: str, path: str | Path) -> "RepresentationFrame":
        """Load a named representation from a parquet file."""
        return cls(name=name, frame=pd.read_parquet(path))


def iter_with_progress(
    iterable: Iterable[T],
    *,
    enabled: bool = False,
    desc: str = "progress",
    total: int | None = None,
    every: int | None = None,
) -> Iterator[T]:
    """Yield items while printing lightweight progress updates when enabled."""
    if not enabled:
        yield from iterable
        return

    if total is None and hasattr(iterable, "__len__"):
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None

    if total is None:
        every = max(1, every or 100)
        print(f"[{desc}] started")
        for idx, item in enumerate(iterable, start=1):
            yield item
            if idx == 1 or idx % every == 0:
                print(f"[{desc}] {idx} items")
        print(f"[{desc}] done")
        return

    every = max(1, every or max(1, total // 20))
    print(f"[{desc}] 0/{total} (0%)")
    for idx, item in enumerate(iterable, start=1):
        yield item
        if idx == 1 or idx == total or idx % every == 0:
            pct = int(round((100.0 * idx) / max(1, total)))
            filled = min(20, int(round(pct / 5)))
            bar = "#" * filled + "-" * (20 - filled)
            print(f"[{desc}] {idx}/{total} ({pct}%) [{bar}]")
    print(f"[{desc}] done")


def binary_log_loss(y_true: np.ndarray, p_pred: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Compute per-row binary log loss from labels and probabilities."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_pred, dtype=float), eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def life_progress(timestamp: pd.Timestamp, created_at: pd.Timestamp, end_date: pd.Timestamp) -> float:
    """Compute the normalized progress of a market life cycle at a timestamp."""
    total = (end_date - created_at).total_seconds()
    elapsed = (timestamp - created_at).total_seconds()
    if total <= 0:
        return np.nan
    return float(np.clip(elapsed / total, 0.0, 1.0))


def default_feature_columns(df: pd.DataFrame, *, exclude: Iterable[str] | None = None) -> list[str]:
    """Return numeric feature columns after removing metadata and target columns."""
    excluded = {
        "market_id",
        "market_slug",
        "question",
        "end_date",
        "created_at",
        "cutoff_timestamp_utc",
        "timestamp_utc",
        "target",
        "future_move",
        "market_price_baseline",
        "market_abs_error",
        "market_log_loss",
        "final_outcome",
        "resolution_source",
        "description",
        "tag_labels",
        "platform_category",
        "research_category",
        "family_id",
    }
    if exclude is not None:
        excluded.update(exclude)

    return sorted(
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    )


def add_time_progress_features(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    created_col: str = "created_at",
    end_col: str = "end_date",
) -> pd.DataFrame:
    """Attach hours-to-resolution, age, and normalized life-progress features."""
    out = df.copy()
    out["hours_to_resolution"] = (
        (pd.to_datetime(out[end_col], utc=True) - pd.to_datetime(out[timestamp_col], utc=True)).dt.total_seconds() / 3600.0
    )
    out["market_age_hours"] = (
        (pd.to_datetime(out[timestamp_col], utc=True) - pd.to_datetime(out[created_col], utc=True)).dt.total_seconds() / 3600.0
    )
    out["life_progress"] = [
        life_progress(ts, created_at, end_date)
        for ts, created_at, end_date in zip(
            pd.to_datetime(out[timestamp_col], utc=True),
            pd.to_datetime(out[created_col], utc=True),
            pd.to_datetime(out[end_col], utc=True),
            strict=False,
        )
    ]
    return out


def safe_float(value: object) -> float:
    """Convert a scalar to float while preserving missing values as NaN."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def safe_mean(values: np.ndarray) -> float:
    """Return a stable mean for possibly empty arrays."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def safe_std(values: np.ndarray) -> float:
    """Return a stable standard deviation for possibly empty arrays."""
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


def safe_sum(values: np.ndarray) -> float:
    """Return a stable sum for possibly empty arrays."""
    if len(values) == 0:
        return 0.0
    return float(np.sum(values))


def safe_max(values: np.ndarray) -> float:
    """Return a stable max for possibly empty arrays."""
    if len(values) == 0:
        return 0.0
    return float(np.max(values))
