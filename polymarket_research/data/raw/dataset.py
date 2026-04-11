"""Raw source-of-truth dataset objects for Polymarket export tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from polymarket_research.utils.data import (
    default_db_path,
    load_selected_markets,
    load_probabilities_for_market_frame,
    open_sqlite_dataset,
    resolve_repo_root,
)


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


@dataclass
class RawPolymarketDataset:
    """Load, hold, and persist raw Polymarket export tables for research use."""

    db_path: Path = field(default_factory=default_db_path)
    market_universe: pd.DataFrame | None = None
    selected_markets: pd.DataFrame | None = None
    added_markets: pd.DataFrame | None = None
    probabilities: pd.DataFrame | None = None
    raw_trades: pd.DataFrame | None = None

    @property
    def markets(self) -> pd.DataFrame | None:
        """Backward-compatible alias for the selected market registry."""
        return self.selected_markets

    @markets.setter
    def markets(self, value: pd.DataFrame | None) -> None:
        self.selected_markets = value

    def load_selected_markets(self) -> pd.DataFrame:
        """Load the full selected market registry from SQLite."""
        with open_sqlite_dataset(self.db_path) as conn:
            self.selected_markets = load_selected_markets(conn)
        return self.selected_markets

    def load_market_universe(self) -> pd.DataFrame:
        """Load raw market-universe rows for the currently selected market ids."""
        if self.selected_markets is None:
            self.load_selected_markets()
        assert self.selected_markets is not None

        with open_sqlite_dataset(self.db_path) as conn:
            self.market_universe = _load_table_for_market_ids(
                conn,
                table_name="market_universe",
                market_ids=self.selected_markets["market_id"].tolist(),
                order_by="created_at DESC",
            )
        return self.market_universe

    def load_added_markets(self) -> pd.DataFrame:
        """Load the added-markets manifest rows for the current selected scope."""
        if self.selected_markets is None:
            self.load_selected_markets()
        assert self.selected_markets is not None

        with open_sqlite_dataset(self.db_path) as conn:
            self.added_markets = _load_table_for_market_ids(
                conn,
                table_name="added_markets",
                market_ids=self.selected_markets["market_id"].tolist(),
                order_by="added_at_utc DESC",
            )
        return self.added_markets

    def load_probabilities(self) -> pd.DataFrame:
        """Load raw probability history for the currently selected markets."""
        if self.selected_markets is None:
            self.load_selected_markets()
        assert self.selected_markets is not None

        with open_sqlite_dataset(self.db_path) as conn:
            self.probabilities = load_probabilities_for_market_frame(conn, self.selected_markets)
        return self.probabilities

    def load_raw_trades(self) -> pd.DataFrame:
        """Load raw normalized fill rows for the currently selected markets."""
        if self.selected_markets is None:
            self.load_selected_markets()
        assert self.selected_markets is not None

        with open_sqlite_dataset(self.db_path) as conn:
            self.raw_trades = _load_table_for_market_ids(
                conn,
                table_name="raw_trades",
                market_ids=self.selected_markets["market_id"].tolist(),
                order_by="market_id, timestamp_utc",
            )
        return self.raw_trades

    def load(self, *, include_raw_trades: bool = False) -> "RawPolymarketDataset":
        """Load the raw export tables for the configured selected-market scope."""
        self.load_selected_markets()
        self.load_market_universe()
        self.load_added_markets()
        self.load_probabilities()
        if include_raw_trades:
            self.load_raw_trades()
        return self

    @property
    def is_loaded(self) -> bool:
        """Return whether the core raw export tables are loaded in memory."""
        return (
            self.market_universe is not None
            and self.selected_markets is not None
            and self.added_markets is not None
            and self.probabilities is not None
        )

    @property
    def short_markets(self) -> pd.DataFrame:
        """Return a compact selected-market view for quick inspection."""
        columns = [
            "market_id",
            "event_id",
            "event_slug",
            "event_title",
            "event_series_slug",
            "question",
            "primary_domain",
            "active",
            "closed",
            "resolved",
            "probability_start_utc",
            "probability_end_utc",
            "raw_trade_rows",
            "raw_trades_saved",
            "final_outcome",
        ]
        if self.selected_markets is None:
            return pd.DataFrame(columns=columns)
        available = [column for column in columns if column in self.selected_markets.columns]
        return self.selected_markets[available].copy()

    @property
    def short_raw_trades(self) -> pd.DataFrame:
        """Return a compact raw-trades view for quick inspection."""
        columns = [
            "trade_id",
            "market_id",
            "condition_id",
            "asset_id",
            "timestamp_utc",
            "price",
            "size",
            "outcome",
            "maker",
            "taker",
            "fee",
        ]
        if self.raw_trades is None:
            return pd.DataFrame(columns=columns)
        available = [column for column in columns if column in self.raw_trades.columns]
        return self.raw_trades[available].copy()

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the currently loaded raw tables."""
        tables = [
            ("market_universe", self.market_universe),
            ("selected_markets", self.selected_markets),
            ("added_markets", self.added_markets),
            ("probabilities", self.probabilities),
            ("raw_trades", self.raw_trades),
        ]
        return pd.DataFrame(
            [
                {
                    "name": name,
                    "loaded": frame is not None,
                    "rows": 0 if frame is None else len(frame),
                    "cols": 0 if frame is None else frame.shape[1],
                }
                for name, frame in tables
            ]
        )

    def save(self, directory: str | Path) -> pd.DataFrame:
        """Save loaded raw tables as parquet files."""
        if not self.is_loaded:
            raise RuntimeError("Core raw tables must be loaded before save().")

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[tuple[str, pd.DataFrame | None]] = [
            ("market_universe.parquet", self.market_universe),
            ("selected_markets.parquet", self.selected_markets),
            ("markets.parquet", self.selected_markets),
            ("added_markets.parquet", self.added_markets),
            ("probabilities.parquet", self.probabilities),
            ("raw_trades.parquet", self.raw_trades),
        ]
        manifest_rows: list[dict[str, object]] = []
        for filename, frame in outputs:
            if frame is None:
                continue
            frame.to_parquet(target_dir / filename, index=False)
            manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
        return pd.DataFrame(manifest_rows)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "RawPolymarketDataset":
        """Instantiate the raw dataset object from saved parquet files."""
        source_dir = Path(directory)
        instance = cls()

        selected_path = source_dir / "selected_markets.parquet"
        legacy_markets_path = source_dir / "markets.parquet"
        market_universe_path = source_dir / "market_universe.parquet"
        added_markets_path = source_dir / "added_markets.parquet"
        probabilities_path = source_dir / "probabilities.parquet"
        raw_trades_path = source_dir / "raw_trades.parquet"

        instance.selected_markets = (
            pd.read_parquet(selected_path)
            if selected_path.exists()
            else pd.read_parquet(legacy_markets_path)
        )
        instance.market_universe = (
            pd.read_parquet(market_universe_path)
            if market_universe_path.exists()
            else instance.selected_markets.copy()
        )
        instance.added_markets = (
            pd.read_parquet(added_markets_path)
            if added_markets_path.exists()
            else _empty_frame(
                [
                    "market_id",
                    "condition_id",
                    "market_slug",
                    "primary_domain",
                    "added_at_utc",
                    "trade_rows",
                    "probability_rows",
                    "probability_start_utc",
                    "probability_end_utc",
                    "storage_path",
                    "raw_trade_rows",
                    "raw_trade_start_utc",
                    "raw_trade_end_utc",
                    "raw_trades_saved",
                ]
            )
        )
        instance.probabilities = pd.read_parquet(probabilities_path)
        instance.raw_trades = pd.read_parquet(raw_trades_path) if raw_trades_path.exists() else None
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


def _load_table_for_market_ids(
    conn,
    *,
    table_name: str,
    market_ids: list[str],
    order_by: str,
) -> pd.DataFrame:
    market_ids = [str(market_id) for market_id in market_ids]
    if not market_ids:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for chunk_start in range(0, len(market_ids), 250):
        chunk = market_ids[chunk_start : chunk_start + 250]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE market_id IN ({placeholders})
        ORDER BY {order_by}
        """
        frames.append(pd.read_sql_query(query, conn, params=tuple(chunk)))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _normalize_generic_export_frame(out)


def _normalize_generic_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    datetime_columns = [
        "created_at",
        "end_date",
        "synced_at_utc",
        "added_at_utc",
        "probability_start_utc",
        "probability_end_utc",
        "raw_trade_start_utc",
        "raw_trade_end_utc",
        "timestamp_utc",
    ]
    for column in datetime_columns:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")

    numeric_columns = [
        "volume_num",
        "liquidity_num",
        "final_yes_probability",
        "trade_rows",
        "probability_rows",
        "raw_trade_rows",
        "yes_probability",
        "trade_count",
        "total_size",
        "last_trade_price",
        "price",
        "size",
        "fee",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    bool_like_columns = [
        "active",
        "closed",
        "archived",
        "observed_trade",
        "raw_trades_saved",
    ]
    for column in bool_like_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)

    if "market_id" in out.columns:
        out["market_id"] = out["market_id"].astype(str)
    if "condition_id" in out.columns:
        out["condition_id"] = out["condition_id"].astype("string")
    if "event_id" in out.columns:
        out["event_id"] = out["event_id"].astype("string")
    if {"end_date", "synced_at_utc"}.issubset(out.columns):
        out["resolved"] = (
            out["end_date"].notna()
            & out["synced_at_utc"].notna()
            & (out["end_date"] <= out["synced_at_utc"])
        )
    return out.reset_index(drop=True)
