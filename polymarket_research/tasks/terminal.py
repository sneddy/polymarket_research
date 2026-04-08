"""Task definitions for terminal-outcome prediction."""

from __future__ import annotations

from dataclasses import dataclass

from polymarket_research.data.representations.common import RepresentationFrame
from polymarket_research.tasks.base import TaskFrame


@dataclass
class TerminalOutcomeTaskBuilder:
    """Convert a terminal representation panel into a task-ready prediction frame."""

    terminal_panel: RepresentationFrame
    horizon_hours: int | None = None

    def build(self) -> TaskFrame:
        """Build the terminal outcome task at an optional fixed horizon."""
        frame = self.terminal_panel.frame.copy()
        if self.horizon_hours is not None and "horizon_hours" in frame.columns:
            frame = frame.loc[frame["horizon_hours"] == int(self.horizon_hours)].reset_index(drop=True)
        return TaskFrame(
            name="terminal_outcome",
            frame=frame,
            target_col="target",
            time_col="end_date",
            entity_id_col="market_id",
        )
