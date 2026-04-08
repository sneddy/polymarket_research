"""Raw source-of-truth dataset objects for Polymarket data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from polymarket_research.utils.data import (
    default_db_path,
    load_markets_for_domains,
    load_probabilities_for_market_frame,
    load_saved_dataset_frames,
    open_sqlite_dataset,
    resolve_repo_root,
    save_dataset_frames,
)


@dataclass
class RawPolymarketDataset:
    """Load, hold, and persist the raw Polymarket market and probability tables."""

    db_path: Path = field(default_factory=default_db_path)
    domains: tuple[str, ...] = ("politics", "geopolitics", "technology", "finance_economy")
    max_markets_per_domain: int = 120
    min_probability_rows: int = 288
    markets: pd.DataFrame | None = None
    probabilities: pd.DataFrame | None = None

    def load_markets(self) -> pd.DataFrame:
        """Load the raw market table from SQLite using the current selection config."""
        with open_sqlite_dataset(self.db_path) as conn:
            self.markets = load_markets_for_domains(
                conn,
                domains=self.domains,
                max_markets_per_domain=self.max_markets_per_domain,
                min_probability_rows=self.min_probability_rows,
            )
        return self.markets

    def load_probabilities(self) -> pd.DataFrame:
        """Load raw probability history for the currently selected market table."""
        if self.markets is None:
            self.load_markets()
        assert self.markets is not None

        with open_sqlite_dataset(self.db_path) as conn:
            self.probabilities = load_probabilities_for_market_frame(conn, self.markets)
        return self.probabilities

    def load(self) -> "RawPolymarketDataset":
        """Load both raw markets and raw probability history into memory."""
        self.load_markets()
        self.load_probabilities()
        return self

    @property
    def is_loaded(self) -> bool:
        """Return whether both raw tables are available in memory."""
        return self.markets is not None and self.probabilities is not None

    @property
    def short_markets(self) -> pd.DataFrame:
        """Return a compact, human-readable market view for quick inspection."""
        columns = [
            "market_id",
            "question",
            "primary_domain",
            "active",
            "closed",
            "resolved",
            "probability_start_utc",
            "probability_end_utc",
            "final_outcome",
        ]
        if self.markets is None:
            return pd.DataFrame(columns=columns)
        available = [column for column in columns if column in self.markets.columns]
        return self.markets[available].copy()

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the in-memory raw tables."""
        return pd.DataFrame(
            [
                {
                    "name": "markets",
                    "loaded": self.markets is not None,
                    "rows": 0 if self.markets is None else len(self.markets),
                    "cols": 0 if self.markets is None else self.markets.shape[1],
                },
                {
                    "name": "probabilities",
                    "loaded": self.probabilities is not None,
                    "rows": 0 if self.probabilities is None else len(self.probabilities),
                    "cols": 0 if self.probabilities is None else self.probabilities.shape[1],
                },
            ]
        )

    def save(self, directory: str | Path) -> pd.DataFrame:
        """Save the currently loaded raw dataset frames as parquet files."""
        if not self.is_loaded:
            raise RuntimeError("Dataset must be fully loaded before save().")
        assert self.markets is not None and self.probabilities is not None
        return save_dataset_frames(
            directory=directory,
            markets=self.markets,
            probabilities=self.probabilities,
        )

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "RawPolymarketDataset":
        """Instantiate the raw dataset object from previously saved parquet files."""
        markets, probabilities = load_saved_dataset_frames(directory)
        instance = cls()
        instance.markets = markets
        instance.probabilities = probabilities
        return instance


@dataclass
class RawExternalCovariates:
    """Load raw external covariate rows without adding research-specific features."""

    path: Path = field(default_factory=lambda: resolve_repo_root() / "cached_data" / "external_covariates")
    covariates: pd.DataFrame | None = None

    def load(self) -> "RawExternalCovariates":
        """Load raw external covariates from parquet partitions."""
        frames: list[pd.DataFrame] = []
        for file_path in sorted(Path(self.path).rglob("*.parquet")):
            frame = pd.read_parquet(file_path)
            if "series_id" not in frame.columns:
                for parent in file_path.parents:
                    if parent.name.startswith("series_id="):
                        frame["series_id"] = parent.name.split("=", 1)[1]
                        break
            frames.append(frame)
        if frames:
            self.covariates = pd.concat(frames, ignore_index=True)
        else:
            self.covariates = pd.DataFrame(columns=["timestamp_utc", "series_id", "value"])
        return self

    @property
    def is_loaded(self) -> bool:
        """Return whether the raw external covariate table is loaded."""
        return self.covariates is not None

    @property
    def short_covariates(self) -> pd.DataFrame:
        """Return a compact external-covariate view centered on series, time, and value."""
        columns = [
            "series_id",
            "timestamp_utc",
            "close_timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "trade_count",
        ]
        if self.covariates is None:
            return pd.DataFrame(columns=columns)
        available = [column for column in columns if column in self.covariates.columns]
        return self.covariates[available].copy()

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the in-memory external covariate table."""
        return pd.DataFrame(
            [
                {
                    "name": "external_covariates",
                    "loaded": self.covariates is not None,
                    "rows": 0 if self.covariates is None else len(self.covariates),
                    "cols": 0 if self.covariates is None else self.covariates.shape[1],
                }
            ]
        )

    def save(self, path: str | Path) -> Path:
        """Persist the loaded external covariates as a single parquet file."""
        if not self.is_loaded:
            raise RuntimeError("External covariates must be loaded before save().")
        assert self.covariates is not None

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.covariates.to_parquet(out_path, index=False)
        return out_path

    @classmethod
    def from_parquet(cls, path: str | Path) -> "RawExternalCovariates":
        """Instantiate raw external covariates from a saved parquet file."""
        instance = cls(path=Path(path).parent)
        instance.covariates = pd.read_parquet(path)
        return instance
