"""Repository layer for loading resolved Polymarket markets and probability histories."""

from __future__ import annotations

import sqlite3

import pandas as pd

from benchmarks.benchmark_utils import connect, load_eligible_markets, load_probabilities_for_markets
from polymarket_research.data.config import DataPaths, MarketSelectionConfig


class ResolvedMarketRepository:
    """Load resolved markets and their probability trajectories from the local SQLite store."""

    def __init__(self, paths: DataPaths, selection: MarketSelectionConfig) -> None:
        """Store the filesystem and selection config used for all repository reads."""

        self.paths = paths
        self.selection = selection

    def open(self) -> sqlite3.Connection:
        """Open a SQLite connection to the resolved Polymarket database."""

        return connect(self.paths.db_path)

    def load_markets(self) -> pd.DataFrame:
        """Load eligible markets across every configured domain and concatenate them into one frame."""

        with self.open() as conn:
            frames = [
                load_eligible_markets(
                    conn,
                    domain=domain,
                    max_markets=self.selection.max_markets_per_domain,
                    min_probability_rows=self.selection.min_probability_rows,
                )
                for domain in self.selection.domains
            ]
        return pd.concat(frames, ignore_index=True)

    def load_probabilities(self, market_ids: list[str] | pd.Series) -> pd.DataFrame:
        """Load 5-minute probability history for the provided market identifiers."""

        market_ids = [str(market_id) for market_id in market_ids]
        with self.open() as conn:
            return load_probabilities_for_markets(conn, market_ids)
