"""Stable example id helpers for benchmark tasks."""

from __future__ import annotations

import pandas as pd


def build_example_id(frame: pd.DataFrame, *, prefix_parts: list[str]) -> pd.Series:
    """Build a stable example id from selected columns."""
    id_parts: list[pd.Series] = []
    for column in prefix_parts:
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized = pd.to_datetime(series, utc=True, errors="coerce").dt.strftime("%Y%m%dT%H%M%SZ")
        else:
            normalized = series.astype(str)
        id_parts.append(normalized)

    if not id_parts:
        return pd.RangeIndex(start=0, stop=len(frame)).astype(str)

    example_id = id_parts[0]
    for series in id_parts[1:]:
        example_id = example_id + "__" + series
    return example_id


def format_terminal_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical terminal example ids for a frame."""
    return build_example_id(
        frame.assign(horizon_tag="h" + frame["horizon_hours"].astype(int).astype(str)),
        prefix_parts=["market_id", "horizon_tag", "cutoff_timestamp_utc"],
    )


def format_decisiveness_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical decisiveness example ids for a frame."""
    return build_example_id(
        frame.assign(task_tag="decisive"),
        prefix_parts=["market_id", "task_tag", "cutoff_timestamp_utc"],
    )


def format_repricing_example_ids(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical repricing example ids for a frame."""
    return build_example_id(
        frame.assign(horizon_tag="repricing_h" + frame["future_horizon_hours"].astype(int).astype(str)),
        prefix_parts=["market_id", "horizon_tag", "timestamp_utc"],
    )
