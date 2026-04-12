from __future__ import annotations

import sqlite3

import pandas as pd


def _upsert_frame(conn: sqlite3.Connection, *, table_name: str, key_column: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows = [
        tuple(None if pd.isna(value) else value for value in row)
        for row in df.itertuples(index=False, name=None)
    ]
    columns = list(df.columns)
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    update_sql = ", ".join([f"{column} = excluded.{column}" for column in columns if column != key_column])
    sql = f"""
    INSERT INTO {table_name} ({column_sql})
    VALUES ({placeholders})
    ON CONFLICT({key_column}) DO UPDATE SET
        {update_sql}
    """
    with conn:
        conn.executemany(sql, rows)
    return int(len(df))


def upsert_raw_markets(conn: sqlite3.Connection, raw_df: pd.DataFrame) -> int:
    """Upsert the fetched Kalshi raw markets table by `market_id`."""
    return _upsert_frame(conn, table_name="raw_markets", key_column="market_id", df=raw_df)


def upsert_market_universe(conn: sqlite3.Connection, markets_df: pd.DataFrame) -> int:
    """Upsert the fetched Kalshi market universe by `market_id`."""
    return _upsert_frame(conn, table_name="market_universe", key_column="market_id", df=markets_df)


def upsert_event_metadata(conn: sqlite3.Connection, events_df: pd.DataFrame) -> int:
    """Upsert targeted Kalshi event enrichment rows by `event_id`."""
    return _upsert_frame(conn, table_name="event_metadata", key_column="event_id", df=events_df)


def upsert_selected_markets(conn: sqlite3.Connection, selected_df: pd.DataFrame) -> int:
    """Upsert selected Kalshi markets by `market_id`."""
    return _upsert_frame(conn, table_name="selected_markets", key_column="market_id", df=selected_df)
