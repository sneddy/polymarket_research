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
    """Upsert the fetched Kalshi raw markets table by `market_id`, merging live and historical rows."""
    if raw_df.empty:
        return 0

    rows = [
        tuple(None if pd.isna(value) else value for value in row)
        for row in raw_df.itertuples(index=False, name=None)
    ]
    columns = list(raw_df.columns)
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)

    update_parts: list[str] = []
    for column in columns:
        if column == "market_id":
            continue
        if column == "data_source_kind":
            update_parts.append(
                """
                data_source_kind = CASE
                    WHEN raw_markets.data_source_kind IS NULL THEN excluded.data_source_kind
                    WHEN excluded.data_source_kind IS NULL THEN raw_markets.data_source_kind
                    WHEN raw_markets.data_source_kind = excluded.data_source_kind THEN excluded.data_source_kind
                    WHEN (
                        raw_markets.data_source_kind IN ('markets_live_index', 'markets_historical_index', 'markets_live_and_historical_index')
                        AND excluded.data_source_kind IN ('markets_live_index', 'markets_historical_index', 'markets_live_and_historical_index')
                    ) THEN 'markets_live_and_historical_index'
                    ELSE excluded.data_source_kind
                END
                """.strip()
            )
        elif column == "indexed_at_utc":
            update_parts.append("indexed_at_utc = excluded.indexed_at_utc")
        else:
            update_parts.append(f"{column} = COALESCE(excluded.{column}, raw_markets.{column})")

    sql = f"""
    INSERT INTO raw_markets ({column_sql})
    VALUES ({placeholders})
    ON CONFLICT(market_id) DO UPDATE SET
        {", ".join(update_parts)}
    """
    with conn:
        conn.executemany(sql, rows)
    return int(len(raw_df))


def upsert_raw_series(conn: sqlite3.Connection, series_df: pd.DataFrame) -> int:
    """Upsert the fetched Kalshi raw series table by `series_ticker`."""
    return _upsert_frame(conn, table_name="raw_series", key_column="series_ticker", df=series_df)


def upsert_selected_series(conn: sqlite3.Connection, selected_df: pd.DataFrame) -> int:
    """Upsert the selected Kalshi series table by `series_ticker`."""
    return _upsert_frame(conn, table_name="selected_series", key_column="series_ticker", df=selected_df)


def upsert_market_universe(conn: sqlite3.Connection, markets_df: pd.DataFrame) -> int:
    """Upsert the fetched Kalshi market universe by `market_id`."""
    return _upsert_frame(conn, table_name="market_universe", key_column="market_id", df=markets_df)


def upsert_event_metadata(conn: sqlite3.Connection, events_df: pd.DataFrame) -> int:
    """Upsert targeted Kalshi event enrichment rows by `event_id`."""
    return _upsert_frame(conn, table_name="event_metadata", key_column="event_id", df=events_df)


def upsert_selected_markets(conn: sqlite3.Connection, selected_df: pd.DataFrame) -> int:
    """Upsert selected Kalshi markets by `market_id`."""
    return _upsert_frame(conn, table_name="selected_markets", key_column="market_id", df=selected_df)


def _upsert_frame_composite(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    key_columns: list[str],
    df: pd.DataFrame,
) -> int:
    if df.empty:
        return 0

    rows = [
        tuple(None if pd.isna(value) else value for value in row)
        for row in df.itertuples(index=False, name=None)
    ]
    columns = list(df.columns)
    placeholders = ", ".join(["?"] * len(columns))
    column_sql = ", ".join(columns)
    update_sql = ", ".join([f"{column} = excluded.{column}" for column in columns if column not in key_columns])
    conflict_sql = ", ".join(key_columns)
    sql = f"""
    INSERT INTO {table_name} ({column_sql})
    VALUES ({placeholders})
    ON CONFLICT({conflict_sql}) DO UPDATE SET
        {update_sql}
    """
    with conn:
        conn.executemany(sql, rows)
    return int(len(df))


def upsert_probabilities(conn: sqlite3.Connection, probabilities_df: pd.DataFrame) -> int:
    """Upsert Kalshi probability panel rows by `(market_id, timestamp_utc)`."""
    return _upsert_frame_composite(
        conn,
        table_name="probabilities",
        key_columns=["market_id", "timestamp_utc"],
        df=probabilities_df,
    )


def upsert_minute_candles(conn: sqlite3.Connection, minute_candles_df: pd.DataFrame) -> int:
    """Upsert Kalshi minute-candle rows by `(market_id, timestamp_utc)`."""
    return _upsert_frame_composite(
        conn,
        table_name="minute_candles",
        key_columns=["market_id", "timestamp_utc"],
        df=minute_candles_df,
    )


def upsert_added_markets(conn: sqlite3.Connection, added_df: pd.DataFrame) -> int:
    """Upsert Kalshi history-manifest rows by `market_id`."""
    return _upsert_frame(conn, table_name="added_markets", key_column="market_id", df=added_df)
