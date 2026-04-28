"""Timestamp helpers for benchmark schemas and builders."""

from __future__ import annotations

import pandas as pd


def normalize_utc_timestamp(timestamp: pd.Timestamp | None) -> pd.Timestamp | None:
    """Normalize an optional timestamp to UTC."""
    if timestamp is None:
        return None
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
