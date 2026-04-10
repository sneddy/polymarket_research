"""Helpers for incremental market history downloads stored in SQLite."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
import logging
import sqlite3

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _pending_predicate(*, require_raw_trades: bool) -> str:
    if require_raw_trades:
        return "(a.market_id IS NULL OR COALESCE(a.raw_trades_saved, 0) = 0)"
    return "a.market_id IS NULL"


def build_pending_queue(
    conn: sqlite3.Connection,
    category: str,
    *,
    require_raw_trades: bool = False,
) -> pd.DataFrame:
    """Return market rows that still need probability history downloads."""
    return pd.read_sql_query(
        f"""
        SELECT
            m.market_id,
            m.condition_id,
            m.market_slug,
            m.event_id,
            m.event_slug,
            m.event_title,
            m.event_series_slug,
            m.question,
            m.description,
            m.resolution_source,
            m.created_at,
            m.end_date,
            m.volume_num,
            m.final_outcome,
            m.final_yes_probability,
            COALESCE(m.outcomes, u.outcomes) AS outcomes,
            COALESCE(m.clob_token_ids, u.clob_token_ids) AS clob_token_ids,
            m.tag_labels,
            m.matched_tags,
            m.matched_domains,
            m.primary_domain
        FROM selected_markets AS m
        LEFT JOIN market_universe AS u
            ON u.market_id = m.market_id
        LEFT JOIN added_markets AS a
            ON a.market_id = m.market_id
        WHERE m.primary_domain = ?
          AND {_pending_predicate(require_raw_trades=require_raw_trades)}
        ORDER BY m.created_at DESC, m.volume_num DESC
        """,
        conn,
        params=(category,),
    )


def build_pending_queue_all(
    conn: sqlite3.Connection,
    *,
    require_raw_trades: bool = False,
) -> pd.DataFrame:
    """Return all filtered market rows that still need probability history downloads."""
    return pd.read_sql_query(
        f"""
        SELECT
            m.market_id,
            m.condition_id,
            m.market_slug,
            m.event_id,
            m.event_slug,
            m.event_title,
            m.event_series_slug,
            m.question,
            m.description,
            m.resolution_source,
            m.created_at,
            m.end_date,
            m.volume_num,
            m.final_outcome,
            m.final_yes_probability,
            COALESCE(m.outcomes, u.outcomes) AS outcomes,
            COALESCE(m.clob_token_ids, u.clob_token_ids) AS clob_token_ids,
            m.tag_labels,
            m.matched_tags,
            m.matched_domains,
            m.primary_domain
        FROM selected_markets AS m
        LEFT JOIN market_universe AS u
            ON u.market_id = m.market_id
        LEFT JOIN added_markets AS a
            ON a.market_id = m.market_id
        WHERE {_pending_predicate(require_raw_trades=require_raw_trades)}
        ORDER BY m.created_at DESC, m.volume_num DESC
        """,
        conn,
    )


def build_yes_probability_series_5m(trades_df: pd.DataFrame, market_id: str) -> pd.DataFrame:
    """Normalize Yes/No trades into a 5-minute probability panel."""
    work = trades_df.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["size"] = pd.to_numeric(work["size"], errors="coerce")
    work["outcome"] = work["outcome"].astype("string")

    outcome_norm = work["outcome"].str.strip().str.lower()
    work["yes_probability"] = np.where(
        outcome_norm.eq("yes"),
        work["price"],
        np.where(outcome_norm.eq("no"), 1.0 - work["price"], np.nan),
    )
    work = work.dropna(subset=["timestamp_utc", "yes_probability"]).sort_values("timestamp_utc").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No usable Yes/No trade history remained after normalization.")

    agg = (
        work.set_index("timestamp_utc")
        .groupby(pd.Grouper(freq="5min"))
        .agg(
            last_yes_probability=("yes_probability", "last"),
            trade_count=("transaction_hash", "size"),
            total_size=("size", "sum"),
            last_trade_price=("price", "last"),
        )
    )

    grid = pd.date_range(
        start=work["timestamp_utc"].min().floor("5min"),
        end=work["timestamp_utc"].max().ceil("5min"),
        freq="5min",
        tz="UTC",
    )

    panel = pd.DataFrame(index=grid).join(agg, how="left")
    panel.index.name = "timestamp_utc"
    panel["yes_probability"] = panel["last_yes_probability"].ffill()
    panel["trade_count"] = panel["trade_count"].fillna(0).astype(int)
    panel["total_size"] = panel["total_size"].fillna(0.0)
    panel["observed_trade"] = panel["last_yes_probability"].notna().astype(int)
    panel["market_id"] = str(market_id)
    panel = panel.reset_index()
    panel["timestamp_utc"] = pd.to_datetime(panel["timestamp_utc"], utc=True, errors="coerce").dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return panel[
        [
            "market_id",
            "timestamp_utc",
            "yes_probability",
            "observed_trade",
            "trade_count",
            "total_size",
            "last_trade_price",
        ]
    ]


def _prepare_raw_trades_for_storage(
    trades_df: pd.DataFrame,
    *,
    market_id: str,
    condition_id: str,
) -> pd.DataFrame:
    """Prepare normalized raw fills for SQLite storage."""
    raw = trades_df.copy()
    raw["market_id"] = str(market_id)
    raw["condition_id"] = str(condition_id)
    required_columns = {
        "trade_id": pd.NA,
        "asset_id": pd.NA,
        "price": np.nan,
        "size": np.nan,
        "outcome": pd.NA,
        "transaction_hash": pd.NA,
        "maker": pd.NA,
        "taker": pd.NA,
        "order_hash": pd.NA,
        "fee": np.nan,
    }
    for column_name, default_value in required_columns.items():
        if column_name not in raw.columns:
            raw[column_name] = default_value

    raw["timestamp_utc"] = pd.to_datetime(raw["timestamp_utc"], utc=True, errors="coerce")
    for column_name in ("price", "size", "fee"):
        raw[column_name] = pd.to_numeric(raw[column_name], errors="coerce")

    for column_name in (
        "trade_id",
        "asset_id",
        "outcome",
        "transaction_hash",
        "maker",
        "taker",
        "order_hash",
    ):
        raw[column_name] = raw[column_name].astype("string")

    raw = raw.dropna(subset=["trade_id", "asset_id", "timestamp_utc"]).copy()
    raw["timestamp_utc"] = raw["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    raw = raw.sort_values(["timestamp_utc", "trade_id"], ascending=True, kind="stable").reset_index(drop=True)
    return raw[
        [
            "trade_id",
            "market_id",
            "condition_id",
            "asset_id",
            "timestamp_utc",
            "price",
            "size",
            "outcome",
            "transaction_hash",
            "maker",
            "taker",
            "order_hash",
            "fee",
        ]
    ]


def _replace_market_raw_trades(
    conn: sqlite3.Connection,
    *,
    market_row: pd.Series,
    trades_df: pd.DataFrame,
) -> tuple[int, str | None, str | None]:
    """Replace one market's raw fill rows and return storage metadata."""
    market_id = str(market_row["market_id"])
    condition_id = str(market_row["condition_id"])
    raw_trades_df = _prepare_raw_trades_for_storage(trades_df, market_id=market_id, condition_id=condition_id)
    raw_trade_rows = int(len(raw_trades_df))
    raw_trade_start = raw_trades_df["timestamp_utc"].min() if raw_trade_rows else None
    raw_trade_end = raw_trades_df["timestamp_utc"].max() if raw_trade_rows else None

    conn.execute("DELETE FROM raw_trades WHERE market_id = ?", (market_id,))
    if raw_trade_rows:
        raw_trades_df.to_sql(
            "raw_trades",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )
    return raw_trade_rows, raw_trade_start, raw_trade_end


def _load_existing_raw_trade_metadata(
    conn: sqlite3.Connection,
    *,
    market_id: str,
) -> tuple[int, str | None, str | None, int]:
    row = conn.execute(
        """
        SELECT
            raw_trade_rows,
            raw_trade_start_utc,
            raw_trade_end_utc,
            raw_trades_saved
        FROM added_markets
        WHERE market_id = ?
        """,
        (market_id,),
    ).fetchone()
    if row is None:
        return 0, None, None, 0
    raw_trade_rows = int(row[0] or 0)
    raw_trade_start = None if row[1] is None else str(row[1])
    raw_trade_end = None if row[2] is None else str(row[2])
    raw_trades_saved = int(row[3] or 0)
    return raw_trade_rows, raw_trade_start, raw_trade_end, raw_trades_saved


def store_market_dataset(
    conn: sqlite3.Connection,
    *,
    market_row: pd.Series,
    probability_df: pd.DataFrame,
    trade_rows: int,
    storage_path: str,
    trades_df: pd.DataFrame | None = None,
    save_trades: bool = True,
) -> None:
    """Replace one market's probability rows and mark it as downloaded."""
    market_id = str(market_row["market_id"])
    now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    probability_rows = len(probability_df)
    prob_start = probability_df["timestamp_utc"].min() if probability_rows else None
    prob_end = probability_df["timestamp_utc"].max() if probability_rows else None
    if save_trades and trades_df is None:
        raise ValueError("trades_df is required when save_trades=True")

    with conn:
        if save_trades:
            raw_trade_rows, raw_trade_start, raw_trade_end = _replace_market_raw_trades(
                conn,
                market_row=market_row,
                trades_df=trades_df,
            )
            raw_trades_saved = 1
        else:
            raw_trade_rows, raw_trade_start, raw_trade_end, raw_trades_saved = _load_existing_raw_trade_metadata(
                conn,
                market_id=market_id,
            )

        conn.execute("DELETE FROM probabilities WHERE market_id = ?", (market_id,))
        probability_df.to_sql(
            "probabilities",
            conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )
        conn.execute(
            """
            INSERT INTO added_markets (
                market_id,
                condition_id,
                market_slug,
                primary_domain,
                added_at_utc,
                trade_rows,
                probability_rows,
                probability_start_utc,
                probability_end_utc,
                storage_path,
                raw_trade_rows,
                raw_trade_start_utc,
                raw_trade_end_utc,
                raw_trades_saved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                condition_id = excluded.condition_id,
                market_slug = excluded.market_slug,
                primary_domain = excluded.primary_domain,
                added_at_utc = excluded.added_at_utc,
                trade_rows = excluded.trade_rows,
                probability_rows = excluded.probability_rows,
                probability_start_utc = excluded.probability_start_utc,
                probability_end_utc = excluded.probability_end_utc,
                storage_path = excluded.storage_path,
                raw_trade_rows = excluded.raw_trade_rows,
                raw_trade_start_utc = excluded.raw_trade_start_utc,
                raw_trade_end_utc = excluded.raw_trade_end_utc,
                raw_trades_saved = excluded.raw_trades_saved
            """,
            (
                market_id,
                str(market_row["condition_id"]),
                str(market_row["market_slug"]),
                str(market_row["primary_domain"]),
                now_utc,
                int(trade_rows),
                int(probability_rows),
                prob_start,
                prob_end,
                storage_path,
                int(raw_trade_rows),
                raw_trade_start,
                raw_trade_end,
                int(raw_trades_saved),
            ),
        )
        logger.info(
            "stored market dataset | market_id=%s probability_rows=%s raw_trade_rows=%s raw_trades_saved=%s",
            market_id,
            probability_rows,
            raw_trade_rows,
            raw_trades_saved,
        )
