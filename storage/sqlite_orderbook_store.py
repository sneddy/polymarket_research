from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Sequence


@dataclass(frozen=True)
class SqliteOrderBookStore:
    """SQLite storage for periodic order book snapshots."""

    def initialize(self, path: str | Path) -> Path:
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_outcomes (
                    token_id TEXT PRIMARY KEY,
                    source_url TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_slug TEXT NOT NULL,
                    market_id TEXT,
                    market_slug TEXT NOT NULL,
                    market_question TEXT,
                    group_item_title TEXT,
                    condition_id TEXT NOT NULL,
                    outcome_index INTEGER NOT NULL,
                    outcome_name TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    closed INTEGER NOT NULL,
                    archived INTEGER NOT NULL,
                    enable_order_book INTEGER NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS poll_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_url TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    interval_seconds REAL NOT NULL,
                    levels_requested INTEGER NOT NULL,
                    outcomes_expected INTEGER NOT NULL,
                    outcomes_succeeded INTEGER NOT NULL,
                    outcomes_failed INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    poll_cycle_id INTEGER NOT NULL REFERENCES poll_cycles(id) ON DELETE CASCADE,
                    token_id TEXT NOT NULL REFERENCES market_outcomes(token_id),
                    condition_id TEXT NOT NULL,
                    market_slug TEXT NOT NULL,
                    outcome_name TEXT NOT NULL,
                    captured_at_utc TEXT NOT NULL,
                    book_timestamp_ms INTEGER,
                    book_hash TEXT,
                    best_bid REAL,
                    best_ask REAL,
                    last_trade_price REAL,
                    min_order_size REAL,
                    tick_size REAL,
                    bids_count INTEGER NOT NULL,
                    asks_count INTEGER NOT NULL,
                    UNIQUE(poll_cycle_id, token_id)
                );

                CREATE TABLE IF NOT EXISTS orderbook_levels (
                    snapshot_id INTEGER NOT NULL REFERENCES orderbook_snapshots(id) ON DELETE CASCADE,
                    side TEXT NOT NULL CHECK (side IN ('bid', 'ask')),
                    level_index INTEGER NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    PRIMARY KEY (snapshot_id, side, level_index)
                );

                CREATE TABLE IF NOT EXISTS poll_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    poll_cycle_id INTEGER NOT NULL REFERENCES poll_cycles(id) ON DELETE CASCADE,
                    token_id TEXT NOT NULL,
                    condition_id TEXT NOT NULL,
                    market_slug TEXT NOT NULL,
                    outcome_name TEXT NOT NULL,
                    error_message TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_poll_cycles_captured_at
                    ON poll_cycles (captured_at_utc);

                CREATE INDEX IF NOT EXISTS idx_snapshots_token_time
                    ON orderbook_snapshots (token_id, captured_at_utc);

                CREATE INDEX IF NOT EXISTS idx_errors_cycle
                    ON poll_errors (poll_cycle_id);
                """
            )

        return db_path

    def upsert_market_outcomes(self, path: str | Path, outcomes: Sequence[Any]) -> None:
        if not outcomes:
            return

        db_path = self.initialize(path)
        rows = [
            (
                item.token_id,
                item.source_url,
                item.source_kind,
                item.source_slug,
                item.market_id,
                item.market_slug,
                item.market_question,
                item.group_item_title,
                item.condition_id,
                item.outcome_index,
                item.outcome_name,
                int(item.active),
                int(item.closed),
                int(item.archived),
                int(item.enable_order_book),
                item.updated_at_utc,
            )
            for item in outcomes
        ]

        with self._connect(db_path) as conn:
            conn.executemany(
                """
                INSERT INTO market_outcomes (
                    token_id,
                    source_url,
                    source_kind,
                    source_slug,
                    market_id,
                    market_slug,
                    market_question,
                    group_item_title,
                    condition_id,
                    outcome_index,
                    outcome_name,
                    active,
                    closed,
                    archived,
                    enable_order_book,
                    updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET
                    source_url = excluded.source_url,
                    source_kind = excluded.source_kind,
                    source_slug = excluded.source_slug,
                    market_id = excluded.market_id,
                    market_slug = excluded.market_slug,
                    market_question = excluded.market_question,
                    group_item_title = excluded.group_item_title,
                    condition_id = excluded.condition_id,
                    outcome_index = excluded.outcome_index,
                    outcome_name = excluded.outcome_name,
                    active = excluded.active,
                    closed = excluded.closed,
                    archived = excluded.archived,
                    enable_order_book = excluded.enable_order_book,
                    updated_at_utc = excluded.updated_at_utc
                """,
                rows,
            )

    def append_poll_result(self, path: str | Path, result: Any) -> int:
        db_path = self.initialize(path)

        with self._connect(db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO poll_cycles (
                    source_url,
                    captured_at_utc,
                    interval_seconds,
                    levels_requested,
                    outcomes_expected,
                    outcomes_succeeded,
                    outcomes_failed
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source_url,
                    result.captured_at_utc,
                    result.interval_seconds,
                    result.levels_requested,
                    result.outcomes_expected,
                    len(result.snapshots),
                    len(result.errors),
                ),
            )
            poll_cycle_id = int(cursor.lastrowid)

            for snapshot in result.snapshots:
                snap_cursor = conn.execute(
                    """
                    INSERT INTO orderbook_snapshots (
                        poll_cycle_id,
                        token_id,
                        condition_id,
                        market_slug,
                        outcome_name,
                        captured_at_utc,
                        book_timestamp_ms,
                        book_hash,
                        best_bid,
                        best_ask,
                        last_trade_price,
                        min_order_size,
                        tick_size,
                        bids_count,
                        asks_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        poll_cycle_id,
                        snapshot.token_id,
                        snapshot.condition_id,
                        snapshot.market_slug,
                        snapshot.outcome_name,
                        snapshot.captured_at_utc,
                        snapshot.book_timestamp_ms,
                        snapshot.book_hash,
                        snapshot.best_bid,
                        snapshot.best_ask,
                        snapshot.last_trade_price,
                        snapshot.min_order_size,
                        snapshot.tick_size,
                        len(snapshot.bid_levels),
                        len(snapshot.ask_levels),
                    ),
                )
                snapshot_id = int(snap_cursor.lastrowid)

                level_rows = [
                    (snapshot_id, level.side, level.level_index, level.price, level.size)
                    for level in [*snapshot.bid_levels, *snapshot.ask_levels]
                ]
                conn.executemany(
                    """
                    INSERT INTO orderbook_levels (
                        snapshot_id,
                        side,
                        level_index,
                        price,
                        size
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    level_rows,
                )

            error_rows = [
                (
                    poll_cycle_id,
                    error.token_id,
                    error.condition_id,
                    error.market_slug,
                    error.outcome_name,
                    error.error_message,
                )
                for error in result.errors
            ]
            if error_rows:
                conn.executemany(
                    """
                    INSERT INTO poll_errors (
                        poll_cycle_id,
                        token_id,
                        condition_id,
                        market_slug,
                        outcome_name,
                        error_message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    error_rows,
                )

        return poll_cycle_id

    def get_counts(self, path: str | Path) -> dict[str, int]:
        db_path = Path(path)
        with self._connect(db_path) as conn:
            poll_cycles = self._count(conn, "poll_cycles")
            snapshots = self._count(conn, "orderbook_snapshots")
            levels = self._count(conn, "orderbook_levels")
            errors = self._count(conn, "poll_errors")
            outcomes = self._count(conn, "market_outcomes")
        return {
            "poll_cycles": poll_cycles,
            "snapshots": snapshots,
            "levels": levels,
            "errors": errors,
            "outcomes": outcomes,
        }

    @staticmethod
    def _count(conn: sqlite3.Connection, table: str) -> int:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return 0 if row is None else int(row[0])

    @staticmethod
    def _connect(path: str | Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        return conn
