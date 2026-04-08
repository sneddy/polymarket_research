"""Container objects for raw and derived Polymarket research datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataBundle:
    """Hold the raw market tables and derived panels produced by phase-one preprocessing."""

    markets: pd.DataFrame
    probabilities: pd.DataFrame
    terminal: pd.DataFrame
    repricing: pd.DataFrame
    shock_table: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of bundle shapes for quick inspection."""

        return pd.DataFrame(
            [
                {"name": "markets", "rows": len(self.markets), "cols": self.markets.shape[1]},
                {"name": "probabilities", "rows": len(self.probabilities), "cols": self.probabilities.shape[1]},
                {"name": "terminal", "rows": len(self.terminal), "cols": self.terminal.shape[1]},
                {"name": "repricing", "rows": len(self.repricing), "cols": self.repricing.shape[1]},
                {"name": "shock_table", "rows": len(self.shock_table), "cols": self.shock_table.shape[1]},
            ]
        )

    def save(self, directory: str | Path) -> pd.DataFrame:
        """Persist every dataframe in the bundle as parquet files and return a save manifest."""

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs = [
            ("markets.parquet", self.markets),
            ("probabilities.parquet", self.probabilities),
            ("terminal.parquet", self.terminal),
            ("repricing.parquet", self.repricing),
            ("shock_table.parquet", self.shock_table),
        ]

        manifest_rows: list[dict[str, object]] = []
        for filename, frame in outputs:
            frame.to_parquet(target_dir / filename, index=False)
            manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
        return pd.DataFrame(manifest_rows)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "DataBundle":
        """Instantiate a saved phase-one bundle from parquet files."""
        source_dir = Path(directory)
        return cls(
            markets=pd.read_parquet(source_dir / "markets.parquet"),
            probabilities=pd.read_parquet(source_dir / "probabilities.parquet"),
            terminal=pd.read_parquet(source_dir / "terminal.parquet"),
            repricing=pd.read_parquet(source_dir / "repricing.parquet"),
            shock_table=pd.read_parquet(source_dir / "shock_table.parquet"),
        )
