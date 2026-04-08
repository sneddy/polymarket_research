"""Task definitions for trustworthiness and selective prediction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from polymarket_research.data.representations.common import RepresentationFrame
from polymarket_research.tasks.base import TaskFrame


@dataclass
class TrustTaskBuilder:
    """Build trust tasks from terminal panels using explicit trust targets."""

    terminal_panel: RepresentationFrame
    horizon_hours: int = 24
    mode: str = "error_regression"
    error_threshold: float | None = None

    def build(self) -> TaskFrame:
        """Build either a regression or binary trust task at a fixed horizon."""
        frame = self.terminal_panel.frame.copy()
        frame = frame.loc[frame["horizon_hours"] == int(self.horizon_hours)].reset_index(drop=True)

        if self.mode == "error_regression":
            target_col = "market_abs_error"
        elif self.mode == "binary_bad_state":
            threshold = self.error_threshold
            if threshold is None:
                threshold = float(frame["market_abs_error"].quantile(0.8)) if not frame.empty else 0.2
            frame["bad_state_target"] = (frame["market_abs_error"] >= float(threshold)).astype(int)
            target_col = "bad_state_target"
        else:
            raise ValueError(f"Unknown trust task mode: {self.mode!r}")

        return TaskFrame(
            name=f"trust_{self.mode}",
            frame=frame,
            target_col=target_col,
            time_col="end_date",
            entity_id_col="market_id",
        )
