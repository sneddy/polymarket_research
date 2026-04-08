"""Schema helpers for market metadata storage."""

from __future__ import annotations

import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the SQLite tables required by export pipelines."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL UNIQUE,
            market_slug TEXT NOT NULL,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            event_series_slug TEXT,
            question TEXT NOT NULL,
            description TEXT,
            resolution_source TEXT,
            created_at TEXT,
            end_date TEXT,
            active INTEGER NOT NULL DEFAULT 0,
            closed INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            volume_num REAL,
            liquidity_num REAL,
            final_outcome TEXT NOT NULL,
            final_yes_probability REAL NOT NULL,
            tag_labels TEXT NOT NULL,
            matched_tags TEXT NOT NULL,
            matched_domains TEXT NOT NULL,
            primary_domain TEXT NOT NULL,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS market_universe (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT,
            market_slug TEXT,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            event_series_slug TEXT,
            question TEXT,
            description TEXT,
            resolution_source TEXT,
            created_at TEXT,
            end_date TEXT,
            active INTEGER,
            closed INTEGER,
            archived INTEGER,
            volume_num REAL,
            liquidity_num REAL,
            final_outcome TEXT,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS added_markets (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            primary_domain TEXT NOT NULL,
            added_at_utc TEXT NOT NULL,
            trade_rows INTEGER NOT NULL,
            probability_rows INTEGER NOT NULL,
            probability_start_utc TEXT,
            probability_end_utc TEXT,
            storage_path TEXT NOT NULL,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        );

        CREATE TABLE IF NOT EXISTS probabilities (
            market_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            yes_probability REAL NOT NULL,
            observed_trade INTEGER NOT NULL,
            trade_count INTEGER NOT NULL,
            total_size REAL NOT NULL,
            last_trade_price REAL,
            PRIMARY KEY (market_id, timestamp_utc),
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        );

        CREATE INDEX IF NOT EXISTS idx_markets_primary_domain
            ON markets(primary_domain, volume_num DESC, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_market_universe_created_at
            ON market_universe(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_probabilities_timestamp
            ON probabilities(timestamp_utc);
        """
    )
    existing_market_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(markets)").fetchall()
    }
    for column_name in ("event_id", "event_slug", "event_title", "event_series_slug"):
        if column_name not in existing_market_columns:
            conn.execute(f"ALTER TABLE markets ADD COLUMN {column_name} TEXT")
    existing_universe_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(market_universe)").fetchall()
    }
    for column_name in ("event_id", "event_slug", "event_title", "event_series_slug", "final_outcome"):
        if column_name not in existing_universe_columns:
            conn.execute(f"ALTER TABLE market_universe ADD COLUMN {column_name} TEXT")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a SQLite table exists in the current database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None
