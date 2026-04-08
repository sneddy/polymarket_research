"""Task definitions for future repricing prediction."""

from __future__ import annotations

from dataclasses import dataclass

from polymarket_research.data.representations.common import RepresentationFrame
from polymarket_research.tasks.base import TaskFrame


@dataclass
class RepricingTaskBuilder:
    """Convert a repricing representation panel into a task-ready prediction frame."""

    repricing_panel: RepresentationFrame

    def build(self) -> TaskFrame:
        """Build the repricing task from the repricing representation panel."""
        return TaskFrame(
            name="repricing",
            frame=self.repricing_panel.frame.copy(),
            target_col="target",
            time_col="timestamp_utc",
            entity_id_col="market_id",
        )
