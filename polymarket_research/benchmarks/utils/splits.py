"""Split helpers for benchmark materialization and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def assign_time_splits(
    examples: pd.DataFrame,
    *,
    split_on: str,
    valid_columns: set[str],
    split_timestamp_utc: pd.Timestamp | None,
    train_fraction: float,
) -> pd.Series:
    """Assign train/test splits using a single time threshold."""
    if examples.empty:
        return pd.Series(dtype="string")
    if split_on not in valid_columns:
        valid = ", ".join(sorted(repr(column) for column in valid_columns))
        raise ValueError(f"split_on must be one of: {valid}.")

    split_time = (
        pd.Timestamp(split_timestamp_utc)
        if split_timestamp_utc is not None
        else pd.to_datetime(examples[split_on], utc=True).quantile(float(train_fraction))
    )
    split_source = pd.to_datetime(examples[split_on], utc=True)
    return pd.Series(
        np.where(split_source < split_time, "train", "test"),
        index=examples.index,
        dtype="string",
    )


def assign_group_time_splits(
    examples: pd.DataFrame,
    *,
    group_col: str,
    split_on: str,
    valid_columns: set[str],
    split_timestamp_utc: pd.Timestamp | None,
    train_fraction: float,
    group_timestamp_agg: str = "min",
) -> pd.Series:
    """Assign train/test splits at a group level using one representative timestamp per group."""
    if examples.empty:
        return pd.Series(dtype="string")
    if group_col not in examples.columns:
        raise ValueError(f"group_col {group_col!r} is missing from examples.")
    if split_on not in valid_columns:
        valid = ", ".join(sorted(repr(column) for column in valid_columns))
        raise ValueError(f"split_on must be one of: {valid}.")
    if group_timestamp_agg not in {"min", "max"}:
        raise ValueError("group_timestamp_agg must be one of: 'min', 'max'.")

    split_source = examples.loc[:, [group_col, split_on]].copy()
    split_source[split_on] = pd.to_datetime(split_source[split_on], utc=True, errors="coerce")
    split_source = split_source.loc[
        split_source[group_col].notna() & split_source[split_on].notna(),
        [group_col, split_on],
    ]
    if split_source.empty:
        return pd.Series(index=examples.index, dtype="string")

    split_source = split_source.groupby(group_col, as_index=False, sort=True)[split_on].agg(group_timestamp_agg)
    split_source["split"] = assign_time_splits(
        split_source,
        split_on=split_on,
        valid_columns=valid_columns,
        split_timestamp_utc=split_timestamp_utc,
        train_fraction=train_fraction,
    )
    split_map = dict(zip(split_source[group_col].astype(str), split_source["split"], strict=False))
    return examples[group_col].astype(str).map(split_map).astype("string")


def select_split_rows(frame: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return a defensive copy of a benchmark frame, optionally filtered by split."""
    if split is None:
        return frame.reset_index(drop=True).copy()
    return frame.loc[frame["split"] == str(split)].reset_index(drop=True).copy()


@dataclass(frozen=True)
class SplitFrame:
    """Simple train/test split views for tabular access."""

    frame: pd.DataFrame
    target_col: str
    feature_columns: list[str]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, self.feature_columns].copy()

    @property
    def y(self) -> pd.Series:
        return self.frame.loc[:, self.target_col].copy()
