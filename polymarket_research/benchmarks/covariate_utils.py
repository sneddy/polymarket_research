from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from storage.parquet_store import ParquetStore


def load_external_covariates(path: str | Path) -> pd.DataFrame:
    store = ParquetStore(frame_type="pandas")
    df = store.load(path, frame_type="pandas")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pandas DataFrame from parquet load.")
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.sort_values(["series_id", "timestamp_utc"], kind="stable").reset_index(drop=True)


def pivot_covariates_to_wide(
    covariates_df: pd.DataFrame,
    *,
    value_col: str = "value",
) -> pd.DataFrame:
    work = covariates_df.copy()
    if value_col not in work.columns:
        raise ValueError(f"value_col={value_col!r} not found")
    work["series_id"] = work["series_id"].astype(str)
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    wide = (
        work.pivot_table(
            index="timestamp_utc",
            columns="series_id",
            values=value_col,
            aggfunc="last",
        )
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide


def add_lagged_covariate_features(
    covariate_wide_df: pd.DataFrame,
    *,
    lags: Sequence[int] = (1,),
    pct_change: bool = True,
) -> pd.DataFrame:
    work = covariate_wide_df.copy().sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    value_cols = [col for col in work.columns if col != "timestamp_utc"]
    out = work[["timestamp_utc"]].copy()

    for col in value_cols:
        series = pd.to_numeric(work[col], errors="coerce")
        out[f"{col}_level"] = series
        for lag in lags:
            lag = int(lag)
            if pct_change:
                denom = series.shift(lag).replace(0.0, np.nan)
                out[f"{col}_ret_lag{lag}"] = (series / denom) - 1.0
            else:
                out[f"{col}_diff_lag{lag}"] = series - series.shift(lag)
    return out


def asof_join_covariates(
    base_df: pd.DataFrame,
    covariate_features_df: pd.DataFrame,
    *,
    base_time_col: str,
    covariate_time_col: str = "timestamp_utc",
    max_age: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    left = base_df.copy().sort_values(base_time_col, kind="stable").reset_index(drop=True)
    right = covariate_features_df.copy().sort_values(covariate_time_col, kind="stable").reset_index(drop=True)
    left[base_time_col] = pd.to_datetime(left[base_time_col], utc=True, errors="coerce")
    right[covariate_time_col] = pd.to_datetime(right[covariate_time_col], utc=True, errors="coerce")

    tolerance = None
    if max_age is not None:
        tolerance = pd.Timedelta(max_age)

    return pd.merge_asof(
        left,
        right,
        left_on=base_time_col,
        right_on=covariate_time_col,
        direction="backward",
        tolerance=tolerance,
    )

