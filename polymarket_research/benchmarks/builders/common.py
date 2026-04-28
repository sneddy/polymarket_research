"""Shared helpers for benchmark build-time and notebook analysis frames."""

from __future__ import annotations

import pandas as pd

from polymarket_research.benchmarks.utils.analysis import confidence_slice


def attach_analysis_columns(
    frame: pd.DataFrame,
    *,
    probability_col: str,
    example_id: pd.Series,
    row_id_col: str,
) -> pd.DataFrame:
    """Attach stable ids plus lightweight audit columns used in local notebooks."""
    out = frame.copy()
    out["example_id"] = pd.Series(example_id, index=out.index).astype(str)
    out[row_id_col] = out["example_id"]

    if probability_col in out.columns:
        out["confidence_slice"] = out[probability_col].map(confidence_slice)
        out["hard_case_10_90"] = out[probability_col].between(0.10, 0.90, inclusive="both")
    else:
        out["confidence_slice"] = "unknown"
        out["hard_case_10_90"] = False
    return out
