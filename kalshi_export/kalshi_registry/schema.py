"""Schema helpers for the series-first Kalshi pipeline."""

from __future__ import annotations

import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_series (
            series_ticker TEXT PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            category TEXT,
            tags_json TEXT,
            frequency TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT,
            close_time TEXT,
            settlement_time TEXT,
            raw_payload_json TEXT,
            synced_at_utc TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS selected_series (
            series_ticker TEXT PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            category TEXT,
            tags_json TEXT,
            frequency TEXT,
            status TEXT,
            selection_reason TEXT,
            selection_version TEXT,
            synced_at_utc TEXT NOT NULL
        );

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
            settlement_ts TEXT,
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
            history_start_utc TEXT,
            history_end_utc TEXT,
            history_ready INTEGER,
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

        CREATE TABLE IF NOT EXISTS minute_candles (
            market_id TEXT NOT NULL,
            source TEXT NOT NULL,
            venue_market_id TEXT,
            timestamp_utc TEXT NOT NULL,
            yes_open_probability REAL,
            yes_high_probability REAL,
            yes_low_probability REAL,
            yes_close_probability REAL,
            yes_mean_probability REAL,
            volume_num REAL,
            open_interest_num REAL,
            PRIMARY KEY (market_id, timestamp_utc)
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

        CREATE INDEX IF NOT EXISTS idx_raw_series_category
            ON raw_series(category);

        CREATE INDEX IF NOT EXISTS idx_selected_series_category
            ON selected_series(category);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_series_ticker
            ON raw_markets(series_ticker);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_created_at
            ON raw_markets(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_close_time
            ON raw_markets(close_time DESC);

        CREATE INDEX IF NOT EXISTS idx_raw_markets_status
            ON raw_markets(status);

        CREATE INDEX IF NOT EXISTS idx_selected_markets_history_ready
            ON selected_markets(history_ready);

        CREATE INDEX IF NOT EXISTS idx_probabilities_timestamp
            ON probabilities(timestamp_utc DESC);

        CREATE INDEX IF NOT EXISTS idx_minute_candles_market_ts
            ON minute_candles(market_id, timestamp_utc DESC);
        """
    )
    conn.execute("DROP TABLE IF EXISTS raw_market_series_downloads")
