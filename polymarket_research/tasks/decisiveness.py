"""Task definitions for durable decisive-belief formation."""

from __future__ import annotations

from dataclasses import dataclass

from polymarket_research.data.representations.common import RepresentationFrame
from polymarket_research.tasks.base import TaskFrame


@dataclass
class DecisivenessTaskBuilder:
    """Build decisive-belief tasks from frozen decisiveness benchmark views."""

    decisiveness_panel: RepresentationFrame
    mode: str = "ordinal_horizon"

    def build(self) -> TaskFrame:
        """Build either the ordinal horizon task or the continuous hours-to-decisive task."""
        frame = self.decisiveness_panel.frame.copy()

        if self.mode == "ordinal_horizon":
            target_col = "label"
        elif self.mode == "hours_regression":
            target_col = "hours_to_decisive"
        else:
            raise ValueError(f"Unknown decisiveness task mode: {self.mode!r}")

        return TaskFrame(
            name=f"decisiveness_{self.mode}",
            frame=frame,
            target_col=target_col,
            time_col="cutoff_timestamp_utc",
            entity_id_col="example_id" if "example_id" in frame.columns else "market_id",
        )
