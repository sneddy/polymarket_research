"""Helpers for incremental market history downloads stored in SQLite."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime
import sqlite3

import numpy as np
import pandas as pd


def build_pending_queue(conn: sqlite3.Connection, category: str) -> pd.DataFrame:
    """Return market rows that still need probability history downloads."""
    return pd.read_sql_query(
        """
        SELECT
            m.market_id,
            m.condition_id,
            m.market_slug,
            m.question,
            m.description,
            m.resolution_source,
            m.created_at,
            m.end_date,
            m.volume_num,
            m.final_outcome,
            m.final_yes_probability,
            m.tag_labels,
            m.matched_tags,
            m.matched_domains,
            m.primary_domain
        FROM markets AS m
        LEFT JOIN added_markets AS a
            ON a.market_id = m.market_id
        WHERE m.primary_domain = ?
          AND a.market_id IS NULL
        ORDER BY m.volume_num DESC, m.created_at DESC
        """,
        conn,
        params=(category,),
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


def store_market_dataset(
    conn: sqlite3.Connection,
    *,
    market_row: pd.Series,
    probability_df: pd.DataFrame,
    trade_rows: int,
    storage_path: str,
) -> None:
    """Replace one market's probability rows and mark it as downloaded."""
    market_id = str(market_row["market_id"])
    now_utc = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    probability_rows = len(probability_df)
    prob_start = probability_df["timestamp_utc"].min() if probability_rows else None
    prob_end = probability_df["timestamp_utc"].max() if probability_rows else None

    with conn:
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
                storage_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                condition_id = excluded.condition_id,
                market_slug = excluded.market_slug,
                primary_domain = excluded.primary_domain,
                added_at_utc = excluded.added_at_utc,
                trade_rows = excluded.trade_rows,
                probability_rows = excluded.probability_rows,
                probability_start_utc = excluded.probability_start_utc,
                probability_end_utc = excluded.probability_end_utc,
                storage_path = excluded.storage_path
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
            ),
        )
