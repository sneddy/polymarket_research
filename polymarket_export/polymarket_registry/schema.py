"""Schema helpers for market metadata storage."""

from __future__ import annotations

import sqlite3


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a SQLite table exists in the current database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _legacy_table_has_foreign_keys(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a legacy snapshot table still declares foreign keys."""
    return bool(conn.execute(f"PRAGMA foreign_key_list({table_name})").fetchall())


def _rebuild_legacy_snapshot_tables_without_foreign_keys(conn: sqlite3.Connection) -> None:
    """Recreate legacy snapshot tables without foreign keys to current selected_markets."""
    conn.execute("PRAGMA foreign_keys=OFF;")
    if _table_exists(conn, "added_markets_old") and _legacy_table_has_foreign_keys(conn, "added_markets_old"):
        conn.executescript(
            """
            ALTER TABLE added_markets_old RENAME TO added_markets_old_fk;
            CREATE TABLE added_markets_old (
                market_id TEXT PRIMARY KEY,
                condition_id TEXT NOT NULL,
                market_slug TEXT NOT NULL,
                primary_domain TEXT NOT NULL,
                added_at_utc TEXT NOT NULL,
                trade_rows INTEGER NOT NULL,
                probability_rows INTEGER NOT NULL,
                probability_start_utc TEXT,
                probability_end_utc TEXT,
                storage_path TEXT NOT NULL
            );
            INSERT INTO added_markets_old (
                market_id,
                condition_id,
                market_slug,
                primary_domain,
                added_at_utc,
                trade_rows,
                probability_rows,
                probability_start_utc,
                probability_end_utc,
                storage_path
            )
            SELECT
                market_id,
                condition_id,
                market_slug,
                primary_domain,
                added_at_utc,
                trade_rows,
                probability_rows,
                probability_start_utc,
                probability_end_utc,
                storage_path
            FROM added_markets_old_fk;
            DROP TABLE added_markets_old_fk;
            """
        )
    if _table_exists(conn, "probabilities_old") and _legacy_table_has_foreign_keys(conn, "probabilities_old"):
        conn.executescript(
            """
            ALTER TABLE probabilities_old RENAME TO probabilities_old_fk;
            CREATE TABLE probabilities_old (
                market_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                yes_probability REAL NOT NULL,
                observed_trade INTEGER NOT NULL,
                trade_count INTEGER NOT NULL,
                total_size REAL NOT NULL,
                last_trade_price REAL,
                PRIMARY KEY (market_id, timestamp_utc)
            );
            INSERT INTO probabilities_old (
                market_id,
                timestamp_utc,
                yes_probability,
                observed_trade,
                trade_count,
                total_size,
                last_trade_price
            )
            SELECT
                market_id,
                timestamp_utc,
                yes_probability,
                observed_trade,
                trade_count,
                total_size,
                last_trade_price
            FROM probabilities_old_fk;
            DROP TABLE probabilities_old_fk;
            CREATE INDEX IF NOT EXISTS idx_probabilities_old_timestamp
                ON probabilities_old(timestamp_utc);
            """
        )
    conn.execute("PRAGMA foreign_keys=ON;")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the SQLite tables required by export pipelines."""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    existing_tables = {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "markets" in existing_tables and "selected_markets" not in existing_tables:
        conn.execute("ALTER TABLE markets RENAME TO selected_markets")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS selected_markets (
            market_id TEXT PRIMARY KEY,
            condition_id TEXT NOT NULL UNIQUE,
            market_slug TEXT NOT NULL,
            event_id TEXT,
            event_slug TEXT,
            event_title TEXT,
            event_series_slug TEXT,
            event_description TEXT,
            event_start_time TEXT,
            event_score TEXT,
            event_period TEXT,
            event_series_id TEXT,
            event_recurrence TEXT,
            event_series_type TEXT,
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
            outcomes TEXT,
            outcome_prices TEXT,
            clob_token_ids TEXT,
            closed_time TEXT,
            uma_resolution_status TEXT,
            neg_risk INTEGER,
            neg_risk_market_id TEXT,
            group_item_title TEXT,
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
            event_description TEXT,
            event_start_time TEXT,
            event_score TEXT,
            event_period TEXT,
            event_series_id TEXT,
            event_recurrence TEXT,
            event_series_type TEXT,
            question TEXT,
            description TEXT,
            resolution_source TEXT,
            created_at TEXT,
            end_date TEXT,
            closed INTEGER,
            archived INTEGER,
            volume_num REAL,
            liquidity_num REAL,
            outcomes TEXT,
            outcome_prices TEXT,
            clob_token_ids TEXT,
            closed_time TEXT,
            uma_resolution_status TEXT,
            neg_risk INTEGER,
            neg_risk_market_id TEXT,
            group_item_title TEXT,
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
            raw_trade_rows INTEGER NOT NULL DEFAULT 0,
            raw_trade_start_utc TEXT,
            raw_trade_end_utc TEXT,
            raw_trades_saved INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (market_id) REFERENCES selected_markets(market_id)
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
            FOREIGN KEY (market_id) REFERENCES selected_markets(market_id)
        );

        CREATE TABLE IF NOT EXISTS raw_trades (
            trade_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            timestamp_utc TEXT NOT NULL,
            price REAL,
            size REAL,
            outcome TEXT,
            transaction_hash TEXT,
            maker TEXT,
            taker TEXT,
            order_hash TEXT,
            fee REAL,
            PRIMARY KEY (market_id, trade_id),
            FOREIGN KEY (market_id) REFERENCES selected_markets(market_id)
        );

        CREATE INDEX IF NOT EXISTS idx_selected_markets_primary_domain
            ON selected_markets(primary_domain, volume_num DESC, created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_market_universe_created_at
            ON market_universe(created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_probabilities_timestamp
            ON probabilities(timestamp_utc);

        CREATE INDEX IF NOT EXISTS idx_raw_trades_market_timestamp
            ON raw_trades(market_id, timestamp_utc);

        CREATE INDEX IF NOT EXISTS idx_raw_trades_timestamp
            ON raw_trades(timestamp_utc);
        """
    )
    existing_market_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(selected_markets)").fetchall()
    }
    selected_market_column_types = {
        "event_id": "TEXT",
        "event_slug": "TEXT",
        "event_title": "TEXT",
        "event_series_slug": "TEXT",
        "event_description": "TEXT",
        "event_start_time": "TEXT",
        "event_score": "TEXT",
        "event_period": "TEXT",
        "event_series_id": "TEXT",
        "event_recurrence": "TEXT",
        "event_series_type": "TEXT",
        "outcomes": "TEXT",
        "outcome_prices": "TEXT",
        "clob_token_ids": "TEXT",
        "closed_time": "TEXT",
        "uma_resolution_status": "TEXT",
        "neg_risk": "INTEGER",
        "neg_risk_market_id": "TEXT",
        "group_item_title": "TEXT",
    }
    for column_name, column_type in selected_market_column_types.items():
        if column_name not in existing_market_columns:
            conn.execute(f"ALTER TABLE selected_markets ADD COLUMN {column_name} {column_type}")
    existing_added_market_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(added_markets)").fetchall()
    }
    added_market_column_types = {
        "raw_trade_rows": "INTEGER NOT NULL DEFAULT 0",
        "raw_trade_start_utc": "TEXT",
        "raw_trade_end_utc": "TEXT",
        "raw_trades_saved": "INTEGER NOT NULL DEFAULT 0",
    }
    for column_name, column_type in added_market_column_types.items():
        if column_name not in existing_added_market_columns:
            conn.execute(f"ALTER TABLE added_markets ADD COLUMN {column_name} {column_type}")
    conn.execute("DROP INDEX IF EXISTS idx_markets_primary_domain")
    existing_universe_columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(market_universe)").fetchall()
    }
    universe_column_types = {
        "event_id": "TEXT",
        "event_slug": "TEXT",
        "event_title": "TEXT",
        "event_series_slug": "TEXT",
        "event_description": "TEXT",
        "event_start_time": "TEXT",
        "event_score": "TEXT",
        "event_period": "TEXT",
        "event_series_id": "TEXT",
        "event_recurrence": "TEXT",
        "event_series_type": "TEXT",
        "outcomes": "TEXT",
        "outcome_prices": "TEXT",
        "clob_token_ids": "TEXT",
        "closed_time": "TEXT",
        "uma_resolution_status": "TEXT",
        "neg_risk": "INTEGER",
        "neg_risk_market_id": "TEXT",
        "group_item_title": "TEXT",
    }
    for column_name, column_type in universe_column_types.items():
        if column_name not in existing_universe_columns:
            conn.execute(f"ALTER TABLE market_universe ADD COLUMN {column_name} {column_type}")
    _rebuild_legacy_snapshot_tables_without_foreign_keys(conn)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a SQLite table exists in the current database."""
    return _table_exists(conn, table_name)
