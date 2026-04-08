"""External covariate representations and time-safe joins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_research.data.canonical.dataset import CanonicalDataset
from polymarket_research.data.representations.common import RepresentationFrame


def pivot_covariates_to_wide(covariates_df: pd.DataFrame, *, value_col: str = "value") -> pd.DataFrame:
    """Pivot canonical external covariates into a time-indexed wide table."""
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
    lags: tuple[int, ...] = (1,),
    pct_change: bool = True,
) -> pd.DataFrame:
    """Derive simple lagged returns or differences from a wide covariate panel."""
    work = covariate_wide_df.copy().sort_values("timestamp_utc", kind="stable").reset_index(drop=True)
    value_cols = [column for column in work.columns if column != "timestamp_utc"]
    out = work[["timestamp_utc"]].copy()

    for column in value_cols:
        series = pd.to_numeric(work[column], errors="coerce")
        out[f"{column}_level"] = series
        for lag in lags:
            lag = int(lag)
            if pct_change:
                denom = series.shift(lag).replace(0.0, np.nan)
                out[f"{column}_ret_lag{lag}"] = (series / denom) - 1.0
            else:
                out[f"{column}_diff_lag{lag}"] = series - series.shift(lag)
    return out


def asof_join_covariates(
    base_df: pd.DataFrame,
    covariate_features_df: pd.DataFrame,
    *,
    base_time_col: str,
    covariate_time_col: str = "timestamp_utc",
    max_age: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    """Join external features onto a base panel using backward as-of matching."""
    left = base_df.copy().sort_values(base_time_col, kind="stable").reset_index(drop=True)
    right = covariate_features_df.copy().sort_values(covariate_time_col, kind="stable").reset_index(drop=True)
    left[base_time_col] = pd.to_datetime(left[base_time_col], utc=True, errors="coerce")
    right[covariate_time_col] = pd.to_datetime(right[covariate_time_col], utc=True, errors="coerce")
    tolerance = pd.Timedelta(max_age) if max_age is not None else None
    return pd.merge_asof(
        left,
        right,
        left_on=base_time_col,
        right_on=covariate_time_col,
        direction="backward",
        tolerance=tolerance,
    )


@dataclass
class ShockPanelBuilder:
    """Build an event-like external shock panel from canonical covariates."""

    canonical: CanonicalDataset
    z_threshold: float = 2.0
    std_window: int = 288

    def build(self) -> RepresentationFrame:
        """Convert external covariates into returns, z-scores, and shock flags."""
        covariates = self.canonical.external_covariates
        if covariates is None or covariates.empty:
            return RepresentationFrame(name="shock_panel", frame=pd.DataFrame(columns=["timestamp_utc"]))

        wide = pivot_covariates_to_wide(covariates, value_col="value").sort_values("timestamp_utc").reset_index(drop=True)
        out = wide[["timestamp_utc"]].copy()
        value_columns = [column for column in wide.columns if column != "timestamp_utc"]
        for column in value_columns:
            series = pd.to_numeric(wide[column], errors="coerce")
            returns = series.pct_change()
            sigma = returns.rolling(self.std_window, min_periods=max(24, self.std_window // 6)).std()
            z_score = returns / sigma.replace(0.0, np.nan)
            out[f"{column}_ret"] = returns
            out[f"{column}_z"] = z_score
            out[f"{column}_shock"] = (z_score.abs() >= self.z_threshold).astype(float)

        shock_columns = [column for column in out.columns if column.endswith("_shock")]
        out["any_external_shock"] = out[shock_columns].max(axis=1) if shock_columns else 0.0
        return RepresentationFrame(name="shock_panel", frame=out)
