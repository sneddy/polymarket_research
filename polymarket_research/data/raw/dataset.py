"""Raw source-of-truth dataset objects for market export tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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


def _apply_market_selection(
    selected_markets: pd.DataFrame,
    *,
    market_limit: int | None = None,
    market_order: Literal["latest", "largest"] | None = None,
) -> pd.DataFrame:
    """Apply a stable market ordering and optional limit to the selected-market registry."""
    if selected_markets.empty:
        return selected_markets

    order = (market_order or "latest").strip().lower() if market_order is not None else None
    out = selected_markets.copy()

    if order is None or order == "latest":
        # Base loader already returns created_at DESC, volume_num DESC, but keep this explicit.
        sort_columns = [column for column in ("created_at", "volume_num") if column in out.columns]
        ascending = [False] * len(sort_columns)
        if sort_columns:
            out = out.sort_values(sort_columns, ascending=ascending, kind="stable")
    elif order == "largest":
        sort_columns = [column for column in ("volume_num", "created_at") if column in out.columns]
        ascending = [False] * len(sort_columns)
        if sort_columns:
            out = out.sort_values(sort_columns, ascending=ascending, kind="stable")
    else:
        raise ValueError("market_order must be one of: None, 'latest', 'largest'.")

    if market_limit is not None:
        out = out.head(int(market_limit))
    return out.reset_index(drop=True)


@dataclass
class RawMarketBundle:
    """Materialized raw export tables held in memory for research use."""

    market_universe: pd.DataFrame | None = None
    selected_markets: pd.DataFrame | None = None
    added_markets: pd.DataFrame | None = None
    probabilities: pd.DataFrame | None = None
    raw_trades: pd.DataFrame | None = None
    source: str | None = None

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
            "source",
            "event_id",
            "event_slug",
            "event_title",
            "event_series_slug",
            "question",
            "platform_category",
            "research_category",
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

        return RawMarketSnapshot(directory).save_bundle(self)

    @classmethod
    def from_parquet(cls, directory: str | Path) -> "RawMarketBundle":
        """Instantiate the raw dataset object from saved parquet files."""
        return RawMarketSnapshot(directory).load_bundle()


@dataclass
class RawMarketHandle:
    """Access raw export tables from SQLite without requiring eager materialization."""

    db_path: Path | str | None = None
    source: Literal["polymarket", "kalshi"] = "polymarket"
    cache_dir: Path | None = None
    chunk_size: int = 250

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path) if self.db_path is not None else default_db_path(source=self.source)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    def load_selected_markets(
        self,
        *,
        market_limit: int | None = None,
        market_order: Literal["latest", "largest"] | None = None,
    ) -> pd.DataFrame:
        """Load the full selected market registry from SQLite."""
        with open_sqlite_dataset(self.db_path, source=self.source) as conn:
            selected_markets = load_selected_markets(conn, source=self.source)
        return _apply_market_selection(
            selected_markets,
            market_limit=market_limit,
            market_order=market_order,
        )

    def load_market_universe(
        self,
        selected_markets: pd.DataFrame | None = None,
        *,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Load raw market-universe rows for the provided or default selected market ids."""
        selected = self._ensure_selected_markets(selected_markets)
        with open_sqlite_dataset(self.db_path, source=self.source) as conn:
            return _load_table_for_market_ids(
                conn,
                table_name="market_universe",
                market_ids=selected["market_id"].tolist(),
                order_by="created_at DESC",
                chunk_size=self.chunk_size,
                show_progress=show_progress,
            )

    def load_added_markets(
        self,
        selected_markets: pd.DataFrame | None = None,
        *,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Load added-markets manifest rows for the provided or default selected scope."""
        selected = self._ensure_selected_markets(selected_markets)
        with open_sqlite_dataset(self.db_path, source=self.source) as conn:
            return _load_table_for_market_ids(
                conn,
                table_name="added_markets",
                market_ids=selected["market_id"].tolist(),
                order_by="added_at_utc DESC",
                chunk_size=self.chunk_size,
                show_progress=show_progress,
            )

    def load_probabilities(
        self,
        selected_markets: pd.DataFrame | None = None,
        *,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Load raw probability history for the provided or default selected markets."""
        selected = self._ensure_selected_markets(selected_markets)
        with open_sqlite_dataset(self.db_path, source=self.source) as conn:
            return load_probabilities_for_market_frame(conn, selected, show_progress=show_progress)

    def load_raw_trades(
        self,
        selected_markets: pd.DataFrame | None = None,
        *,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Load raw normalized fill rows for the provided or default selected markets."""
        selected = self._ensure_selected_markets(selected_markets)
        with open_sqlite_dataset(self.db_path, source=self.source) as conn:
            return _load_table_for_market_ids(
                conn,
                table_name="raw_trades",
                market_ids=selected["market_id"].tolist(),
                order_by="market_id, timestamp_utc",
                chunk_size=self.chunk_size,
                show_progress=show_progress,
            )

    def load_bundle(
        self,
        *,
        include_market_universe: bool = True,
        include_download_manifest: bool = False,
        include_probabilities: bool = True,
        include_raw_trades: bool = False,
        market_limit: int | None = None,
        market_order: Literal["latest", "largest"] | None = None,
        show_progress: bool = False,
    ) -> RawMarketBundle:
        """Materialize a raw bundle with optional large tables."""
        if show_progress:
            print(f"[raw bundle] loading selected markets from: {self.db_path}")
        selected_markets = self.load_selected_markets(
            market_limit=market_limit,
            market_order=market_order,
        )
        if show_progress:
            print(f"[raw bundle] loaded selected markets (rows={len(selected_markets)})")

        market_universe = None
        if include_market_universe:
            if show_progress:
                print("[raw bundle] loading market_universe")
            market_universe = self.load_market_universe(selected_markets, show_progress=show_progress)
            if show_progress:
                print(f"[raw bundle] loaded market_universe (rows={len(market_universe)})")

        added_markets = None
        if include_download_manifest:
            if show_progress:
                print("[raw bundle] loading added_markets")
            added_markets = self.load_added_markets(selected_markets, show_progress=show_progress)
            if show_progress:
                print(f"[raw bundle] loaded added_markets (rows={len(added_markets)})")

        probabilities = None
        if include_probabilities:
            if show_progress:
                print("[raw bundle] loading probabilities")
            probabilities = self.load_probabilities(selected_markets, show_progress=show_progress)
            if show_progress:
                print(f"[raw bundle] loaded probabilities (rows={len(probabilities)})")

        raw_trades = None
        if include_raw_trades:
            if show_progress:
                print("[raw bundle] loading raw_trades")
            raw_trades = self.load_raw_trades(selected_markets, show_progress=show_progress)
            if show_progress:
                print(f"[raw bundle] loaded raw_trades (rows={len(raw_trades)})")
        return RawMarketBundle(
            market_universe=market_universe,
            selected_markets=selected_markets,
            added_markets=added_markets,
            probabilities=probabilities,
            raw_trades=raw_trades,
            source=self.source,
        )

    def snapshot(self, directory: str | Path | None = None) -> "RawMarketSnapshot":
        """Return a snapshot helper for the configured cache directory."""
        target = directory if directory is not None else self.cache_dir
        if target is None:
            raise ValueError("snapshot() requires an explicit directory or cache_dir on the handle.")
        return RawMarketSnapshot(target)

    def _ensure_selected_markets(self, selected_markets: pd.DataFrame | None) -> pd.DataFrame:
        if selected_markets is not None:
            return selected_markets
        return self.load_selected_markets()


@dataclass(frozen=True)
class RawMarketSnapshot:
    """Persist and reload materialized raw bundles as parquet snapshots."""

    directory: Path | str

    @property
    def path(self) -> Path:
        return Path(self.directory)

    def save_bundle(self, bundle: RawMarketBundle) -> pd.DataFrame:
        """Save a materialized raw bundle as parquet files."""
        target_dir = self.path
        target_dir.mkdir(parents=True, exist_ok=True)

        outputs: list[tuple[str, pd.DataFrame | None]] = [
            ("market_universe.parquet", bundle.market_universe),
            ("selected_markets.parquet", bundle.selected_markets),
            ("added_markets.parquet", bundle.added_markets),
            ("probabilities.parquet", bundle.probabilities),
            ("raw_trades.parquet", bundle.raw_trades),
        ]
        manifest_rows: list[dict[str, object]] = []
        for filename, frame in outputs:
            if frame is None:
                continue
            frame.to_parquet(target_dir / filename, index=False)
            manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
        return pd.DataFrame(manifest_rows)

    def load_bundle(
        self,
        *,
        include_download_manifest: bool = False,
        include_raw_trades: bool = True,
        market_limit: int | None = None,
        market_order: Literal["latest", "largest"] | None = None,
    ) -> RawMarketBundle:
        """Load a materialized raw bundle from parquet files."""
        source_dir = self.path

        selected_path = source_dir / "selected_markets.parquet"
        market_universe_path = source_dir / "market_universe.parquet"
        added_markets_path = source_dir / "added_markets.parquet"
        probabilities_path = source_dir / "probabilities.parquet"
        raw_trades_path = source_dir / "raw_trades.parquet"

        selected_markets = pd.read_parquet(selected_path)
        selected_markets = _normalize_generic_export_frame(selected_markets)
        selected_markets = _apply_market_selection(
            selected_markets,
            market_limit=market_limit,
            market_order=market_order,
        )
        market_universe = (
            _normalize_generic_export_frame(pd.read_parquet(market_universe_path))
            if market_universe_path.exists()
            else selected_markets.copy()
        )
        selected_ids = set(selected_markets["market_id"].astype(str).tolist())
        if not market_universe.empty:
            market_universe = market_universe.loc[market_universe["market_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
        added_markets = None
        if include_download_manifest:
            added_markets = (
                _normalize_generic_export_frame(pd.read_parquet(added_markets_path))
                if added_markets_path.exists()
                else _empty_frame(
                    [
                        "market_id",
                        "condition_id",
                        "market_slug",
                        "research_category",
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
            if not added_markets.empty:
                added_markets = added_markets.loc[added_markets["market_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
        probabilities = _normalize_generic_export_frame(pd.read_parquet(probabilities_path)) if probabilities_path.exists() else None
        if probabilities is not None and not probabilities.empty:
            probabilities = probabilities.loc[probabilities["market_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
        raw_trades = _normalize_generic_export_frame(pd.read_parquet(raw_trades_path)) if include_raw_trades and raw_trades_path.exists() else None
        if raw_trades is not None and not raw_trades.empty:
            raw_trades = raw_trades.loc[raw_trades["market_id"].astype(str).isin(selected_ids)].reset_index(drop=True)
        return RawMarketBundle(
            market_universe=market_universe,
            selected_markets=selected_markets,
            added_markets=added_markets,
            probabilities=probabilities,
            raw_trades=raw_trades,
            source=_infer_source_from_frame(selected_markets),
        )

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
    chunk_size: int = 250,
    show_progress: bool = False,
) -> pd.DataFrame:
    market_ids = [str(market_id) for market_id in market_ids]
    if not market_ids:
        return pd.DataFrame()

    chunk_starts = range(0, len(market_ids), chunk_size)
    if show_progress:
        from tqdm.auto import tqdm

        chunk_starts = tqdm(
            chunk_starts,
            total=((len(market_ids) + chunk_size - 1) // chunk_size),
            desc=f"sqlite {table_name}",
            unit="chunk",
        )

    frames: list[pd.DataFrame] = []
    for chunk_start in chunk_starts:
        chunk = market_ids[chunk_start : chunk_start + chunk_size]
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
        "open_time",
        "close_time",
        "settlement_ts",
        "history_start_utc",
        "history_end_utc",
        "updated_at",
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
        "volume_24h_num",
        "open_interest_num",
        "liquidity_num",
        "liquidity_dollars",
        "final_yes_probability",
        "trade_rows",
        "probability_rows",
        "candle_rows_1m",
        "raw_trade_rows",
        "yes_probability",
        "trade_count",
        "total_size",
        "last_trade_price",
        "price",
        "size",
        "fee",
        "yes_price",
        "no_price",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    bool_like_columns = [
        "active",
        "closed",
        "archived",
        "is_active",
        "is_closed",
        "is_resolved",
        "history_ready",
        "is_binary",
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
    if "resolved" in out.columns:
        out["resolved"] = pd.to_numeric(out["resolved"], errors="coerce").fillna(0).astype(int).astype(bool)
    elif "is_resolved" in out.columns:
        out["resolved"] = pd.to_numeric(out["is_resolved"], errors="coerce").fillna(0).astype(int).astype(bool)
    if {"end_date", "synced_at_utc"}.issubset(out.columns):
        out["resolved"] = out.get(
            "resolved",
            (
                out["end_date"].notna()
                & out["synced_at_utc"].notna()
                & (out["end_date"] <= out["synced_at_utc"])
            ),
        )
    if "kalshi_category" in out.columns or "research_category" in out.columns or "primary_domain" in out.columns:
        out = _attach_category_columns(out)
    return out.reset_index(drop=True)


def _attach_category_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    platform = _normalize_category_series(
        out["kalshi_category"] if "kalshi_category" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
    )
    research_seed = _normalize_category_series(
        out["research_category"] if "research_category" in out.columns else (
            out["primary_domain"] if "primary_domain" in out.columns else pd.Series(pd.NA, index=out.index, dtype="string")
        )
    )
    research = research_seed.fillna(platform)
    out["platform_category"] = platform
    out["research_category"] = research
    return out


def _normalize_category_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    invalid = normalized.isna() | normalized.eq("") | normalized.str.lower().isin(
        {"unknown", "unassigned", "none", "<na>", "nan"}
    )
    return normalized.mask(invalid, pd.NA)


def _infer_source_from_frame(frame: pd.DataFrame) -> str | None:
    if "source" in frame.columns and frame["source"].notna().any():
        return str(frame["source"].dropna().astype(str).iloc[0])
    if "kalshi_category" in frame.columns or "venue_market_id" in frame.columns:
        return "kalshi"
    if not frame.empty:
        return "polymarket"
    return None
