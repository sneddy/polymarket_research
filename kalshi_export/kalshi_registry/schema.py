"""Schema helpers for Kalshi metadata storage."""

from __future__ import annotations

import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the SQLite tables required by the initial Kalshi export pipeline."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_markets (
            market_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            event_id TEXT,
            venue_event_id TEXT,
            series_ticker TEXT,
            ticker TEXT,
            event_ticker TEXT,
            title TEXT,
            question TEXT,
            subtitle TEXT,
            yes_sub_title TEXT,
            no_sub_title TEXT,
            market_type TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            open_time TEXT,
            close_time TEXT,
            expected_expiration_time TEXT,
            expiration_time TEXT,
            latest_expiration_time TEXT,
            settlement_ts TEXT,
            last_price_dollars REAL,
            previous_price_dollars REAL,
            yes_bid_dollars REAL,
            yes_ask_dollars REAL,
            no_bid_dollars REAL,
            no_ask_dollars REAL,
            yes_bid_size_fp REAL,
            yes_ask_size_fp REAL,
            volume_num REAL,
            volume_24h_num REAL,
            open_interest_num REAL,
            liquidity_dollars REAL,
            notional_value_dollars REAL,
            response_price_units TEXT,
            price_level_structure TEXT,
            tick_size INTEGER,
            strike_type TEXT,
            floor_strike REAL,
            cap_strike REAL,
            functional_strike TEXT,
            custom_strike_json TEXT,
            mve_collection_ticker TEXT,
            mve_selected_legs_json TEXT,
            rules_primary TEXT,
            rules_secondary TEXT,
            can_close_early INTEGER,
            early_close_condition TEXT,
            is_provisional INTEGER,
            result TEXT,
            settlement_value_dollars REAL,
            description TEXT,
            end_date TEXT,
            final_outcome TEXT,
            final_yes_probability REAL,
            is_binary INTEGER,
            is_resolved INTEGER,
            is_active INTEGER,
            is_closed INTEGER,
            data_source_kind TEXT,
            indexed_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS market_universe (
            market_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            event_id TEXT,
            venue_event_id TEXT,
            series_ticker TEXT,
            ticker TEXT,
            event_ticker TEXT,
            title TEXT,
            question TEXT,
            subtitle TEXT,
            yes_sub_title TEXT,
            no_sub_title TEXT,
            market_type TEXT,
            status TEXT,
            event_title TEXT,
            event_sub_title TEXT,
            kalshi_category TEXT,
            mutually_exclusive INTEGER,
            strike_period TEXT,
            rules_primary TEXT,
            rules_secondary TEXT,
            can_close_early INTEGER,
            early_close_condition TEXT,
            is_provisional INTEGER,
            result TEXT,
            settlement_value_dollars REAL,
            created_at TEXT,
            updated_at TEXT,
            open_time TEXT,
            close_time TEXT,
            expected_expiration_time TEXT,
            expiration_time TEXT,
            latest_expiration_time TEXT,
            settlement_ts TEXT,
            last_price_dollars REAL,
            previous_price_dollars REAL,
            yes_bid_dollars REAL,
            yes_ask_dollars REAL,
            no_bid_dollars REAL,
            no_ask_dollars REAL,
            yes_bid_size_fp REAL,
            yes_ask_size_fp REAL,
            volume_num REAL,
            volume_24h_num REAL,
            open_interest_num REAL,
            liquidity_dollars REAL,
            notional_value_dollars REAL,
            response_price_units TEXT,
            price_level_structure TEXT,
            tick_size INTEGER,
            strike_type TEXT,
            floor_strike REAL,
            cap_strike REAL,
            functional_strike TEXT,
            custom_strike_json TEXT,
            mve_collection_ticker TEXT,
            mve_selected_legs_json TEXT,
            description TEXT,
            end_date TEXT,
            final_outcome TEXT,
            final_yes_probability REAL,
            is_binary INTEGER,
            is_resolved INTEGER,
            is_active INTEGER,
            is_closed INTEGER,
            data_source_kind TEXT,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS event_metadata (
            event_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            venue_event_id TEXT,
            event_ticker TEXT,
            series_ticker TEXT,
            event_title TEXT,
            event_sub_title TEXT,
            kalshi_category TEXT,
            mutually_exclusive INTEGER,
            strike_period TEXT,
            status TEXT,
            created_at TEXT,
            close_time TEXT,
            last_updated_ts TEXT,
            event_url TEXT,
            rules_primary TEXT,
            subtitle TEXT,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS selected_markets (
            market_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            event_id TEXT,
            venue_event_id TEXT,
            series_ticker TEXT,
            ticker TEXT,
            event_ticker TEXT,
            question TEXT,
            description TEXT,
            event_title TEXT,
            kalshi_category TEXT,
            primary_domain TEXT,
            created_at TEXT,
            open_time TEXT,
            close_time TEXT,
            end_date TEXT,
            status TEXT,
            market_type TEXT,
            is_binary INTEGER,
            is_resolved INTEGER,
            is_active INTEGER,
            is_closed INTEGER,
            mutually_exclusive INTEGER,
            strike_type TEXT,
            custom_strike_json TEXT,
            volume_num REAL,
            volume_24h_num REAL,
            open_interest_num REAL,
            liquidity_dollars REAL,
            final_outcome TEXT,
            final_yes_probability REAL,
            rules_primary TEXT,
            rules_secondary TEXT,
            selection_reason TEXT,
            selection_version TEXT,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS added_markets (
            market_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            series_ticker TEXT,
            primary_domain TEXT,
            added_at_utc TEXT NOT NULL,
            storage_path TEXT,
            history_source_mode TEXT,
            probability_rows INTEGER,
            probability_start_utc TEXT,
            probability_end_utc TEXT,
            candle_rows_1m INTEGER,
            raw_trade_rows INTEGER,
            raw_trade_start_utc TEXT,
            raw_trade_end_utc TEXT,
            raw_trades_saved INTEGER,
            cutoff_ts_used TEXT,
            download_warnings_json TEXT
        );

        CREATE TABLE IF NOT EXISTS probabilities (
            market_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            yes_probability REAL,
            observed_trade INTEGER,
            trade_count INTEGER,
            total_size REAL,
            last_trade_price REAL,
            PRIMARY KEY (market_id, timestamp_utc)
        );

        CREATE TABLE IF NOT EXISTS raw_trades (
            trade_id TEXT PRIMARY KEY,
            market_id TEXT NOT NULL,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            timestamp_utc TEXT NOT NULL,
            price REAL,
            size REAL,
            side TEXT,
            yes_price REAL,
            no_price REAL,
            trade_status TEXT,
            raw_payload_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_raw_markets_created_at
            ON raw_markets(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_close_time
            ON raw_markets(close_time DESC);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_event_id
            ON raw_markets(event_id);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_status
            ON raw_markets(status);

        CREATE INDEX IF NOT EXISTS idx_market_universe_created_at
            ON market_universe(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_market_universe_event_id
            ON market_universe(event_id);

        CREATE INDEX IF NOT EXISTS idx_market_universe_series_ticker
            ON market_universe(series_ticker);

        CREATE INDEX IF NOT EXISTS idx_market_universe_category
            ON market_universe(kalshi_category);

        CREATE INDEX IF NOT EXISTS idx_market_universe_status
            ON market_universe(status);

        CREATE INDEX IF NOT EXISTS idx_event_metadata_series_ticker
            ON event_metadata(series_ticker);

        CREATE INDEX IF NOT EXISTS idx_event_metadata_category
            ON event_metadata(kalshi_category);

        CREATE INDEX IF NOT EXISTS idx_selected_markets_event_id
            ON selected_markets(event_id);

        CREATE INDEX IF NOT EXISTS idx_selected_markets_status
            ON selected_markets(status);

        CREATE INDEX IF NOT EXISTS idx_selected_markets_volume
            ON selected_markets(volume_num DESC);

        CREATE INDEX IF NOT EXISTS idx_added_markets_primary_domain
            ON added_markets(primary_domain);

        CREATE INDEX IF NOT EXISTS idx_probabilities_timestamp
            ON probabilities(timestamp_utc DESC);

        CREATE INDEX IF NOT EXISTS idx_raw_trades_market_ts
            ON raw_trades(market_id, timestamp_utc DESC);
        """
    )
    _migrate_markets_index_to_raw_markets(conn)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _migrate_markets_index_to_raw_markets(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "markets_index"):
        return
    raw_count = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
    if raw_count > 0:
        return
    conn.execute("INSERT INTO raw_markets SELECT * FROM markets_index")
