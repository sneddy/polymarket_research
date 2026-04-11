"""Small helper functions for loading and persisting the basic Polymarket dataset."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Sequence

import pandas as pd


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Resolve the repository root by walking upward until the benchmarks directory is found."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "benchmarks").exists():
            return candidate
    raise RuntimeError(f"Could not locate repository root from start={current}")


def default_db_path(repo_root: str | Path | None = None) -> Path:
    """Return the default SQLite path used by the local resolved Polymarket dataset."""

    root = resolve_repo_root(repo_root)
    return root / "db" / "resolved_probability_dataset.sqlite"


def open_sqlite_dataset(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the local SQLite dataset that stores resolved Polymarket markets and probabilities."""

    conn = sqlite3.connect(str(db_path or default_db_path()))
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def load_selected_markets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load the full selected-market registry plus export download metadata."""

    query = """
    SELECT
        m.market_id,
        m.condition_id,
        COALESCE(u.market_slug, m.market_slug) AS market_slug,
        COALESCE(u.event_id, m.event_id) AS event_id,
        COALESCE(u.event_slug, m.event_slug) AS event_slug,
        COALESCE(u.event_title, m.event_title) AS event_title,
        COALESCE(u.event_series_slug, m.event_series_slug) AS event_series_slug,
        COALESCE(u.question, m.question) AS question,
        COALESCE(u.description, m.description) AS description,
        COALESCE(u.resolution_source, m.resolution_source) AS resolution_source,
        COALESCE(m.active, 0) AS active,
        COALESCE(u.closed, m.closed, 0) AS closed,
        COALESCE(u.archived, m.archived, 0) AS archived,
        COALESCE(u.created_at, m.created_at) AS created_at,
        COALESCE(u.end_date, m.end_date) AS end_date,
        COALESCE(u.volume_num, m.volume_num) AS volume_num,
        COALESCE(u.liquidity_num, m.liquidity_num) AS liquidity_num,
        m.final_outcome,
        m.final_yes_probability,
        m.tag_labels,
        m.matched_tags,
        m.matched_domains,
        m.primary_domain,
        COALESCE(u.synced_at_utc, m.synced_at_utc) AS synced_at_utc,
        a.added_at_utc,
        a.trade_rows,
        a.probability_rows,
        a.probability_start_utc,
        a.probability_end_utc,
        a.raw_trade_rows,
        a.raw_trade_start_utc,
        a.raw_trade_end_utc,
        a.raw_trades_saved
    FROM selected_markets AS m
    LEFT JOIN market_universe AS u
        ON u.market_id = m.market_id
    LEFT JOIN added_markets AS a
        ON a.market_id = m.market_id
    ORDER BY COALESCE(u.created_at, m.created_at) DESC, COALESCE(u.volume_num, m.volume_num, 0.0) DESC
    """
    frame = pd.read_sql_query(query, conn)
    return _normalize_market_frame(frame)


def load_probabilities_for_market_frame(
    conn: sqlite3.Connection,
    markets: pd.DataFrame,
) -> pd.DataFrame:
    """Load probability history for all market ids contained in the provided dataframe."""

    return _load_probabilities_for_markets(conn, markets["market_id"].tolist())


def save_dataset_frames(
    *,
    directory: str | Path,
    markets: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    """Persist the core market and probability tables as parquet files and return a manifest."""

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        ("markets.parquet", markets),
        ("probabilities.parquet", probabilities),
    ]
    manifest_rows: list[dict[str, object]] = []
    for filename, frame in outputs:
        frame.to_parquet(target_dir / filename, index=False)
        manifest_rows.append({"file": filename, "rows": len(frame), "cols": frame.shape[1]})
    return pd.DataFrame(manifest_rows)


def load_saved_dataset_frames(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved core dataset parquet files from a directory."""

    source_dir = Path(directory)
    markets = pd.read_parquet(source_dir / "markets.parquet")
    probabilities = pd.read_parquet(source_dir / "probabilities.parquet")
    return markets, probabilities


def _load_probabilities_for_markets(
    conn: sqlite3.Connection,
    market_ids: Sequence[str],
) -> pd.DataFrame:
    """Load probability trajectories for the requested market ids from SQLite."""

    market_ids = [str(market_id) for market_id in market_ids]
    if not market_ids:
        return pd.DataFrame(
            columns=[
                "market_id",
                "timestamp_utc",
                "yes_probability",
                "observed_trade",
                "trade_count",
                "total_size",
                "last_trade_price",
            ]
        )

    frames: list[pd.DataFrame] = []
    for chunk_start in range(0, len(market_ids), 250):
        chunk = market_ids[chunk_start : chunk_start + 250]
        placeholders = ",".join(["?"] * len(chunk))
        query = f"""
        SELECT
            market_id,
            timestamp_utc,
            yes_probability,
            observed_trade,
            trade_count,
            total_size,
            last_trade_price
        FROM probabilities
        WHERE market_id IN ({placeholders})
        ORDER BY market_id, timestamp_utc
        """
        frames.append(pd.read_sql_query(query, conn, params=tuple(chunk)))

    out = pd.concat(frames, ignore_index=True)
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    for column in ("yes_probability", "trade_count", "total_size", "last_trade_price"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["observed_trade"] = pd.to_numeric(out["observed_trade"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["market_id", "timestamp_utc"], kind="stable").reset_index(drop=True)


def _normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize market metadata types after loading them from SQLite."""

    out = frame.copy()
    for column in ("created_at", "end_date", "probability_start_utc", "probability_end_utc", "synced_at_utc"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], utc=True, errors="coerce")
    for column in ("market_id", "event_id"):
        if column in out.columns:
            out[column] = out[column].astype("string")
    numeric_columns = (
        "volume_num",
        "final_yes_probability",
        "trade_rows",
        "probability_rows",
    )
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("active", "closed", "archived"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype(int)
    if {"end_date", "synced_at_utc"}.issubset(out.columns):
        out["resolved"] = (
            out["end_date"].notna()
            & out["synced_at_utc"].notna()
            & (out["end_date"] <= out["synced_at_utc"])
        )
    out["market_id"] = out["market_id"].astype(str)
    return out.reset_index(drop=True)
