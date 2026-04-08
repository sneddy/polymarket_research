"""Base task abstractions for clean benchmark definitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from polymarket_research.data.representations.common import default_feature_columns


@dataclass(frozen=True)
class TaskFrame:
    """Hold a task-ready frame with explicit target and default feature metadata."""

    name: str
    frame: pd.DataFrame
    target_col: str
    time_col: str
    entity_id_col: str = "market_id"

    @property
    def feature_columns(self) -> list[str]:
        """Return default numeric feature columns for the task frame."""
        return default_feature_columns(self.frame, exclude={self.target_col})

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the task frame."""
        positive_rate = None
        if self.target_col in self.frame.columns and pd.api.types.is_numeric_dtype(self.frame[self.target_col]):
            positive_rate = float(self.frame[self.target_col].mean())
        return pd.DataFrame(
            [
                {
                    "name": self.name,
                    "rows": len(self.frame),
                    "cols": self.frame.shape[1],
                    "target_col": self.target_col,
                    "positive_rate": positive_rate,
                }
            ]
        )
