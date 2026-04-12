from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from kalshi_registry.upsert import upsert_selected_markets


logger = logging.getLogger(__name__)

_SELECTED_MARKETS_COLUMNS = [
    "market_id",
    "source",
    "venue_market_id",
    "event_id",
    "venue_event_id",
    "series_ticker",
    "ticker",
    "event_ticker",
    "question",
    "description",
    "event_title",
    "kalshi_category",
    "primary_domain",
    "created_at",
    "open_time",
    "close_time",
    "end_date",
    "status",
    "market_type",
    "is_binary",
    "is_resolved",
    "is_active",
    "is_closed",
    "mutually_exclusive",
    "strike_type",
    "custom_strike_json",
    "volume_num",
    "volume_24h_num",
    "open_interest_num",
    "liquidity_dollars",
    "final_outcome",
    "final_yes_probability",
    "rules_primary",
    "rules_secondary",
    "selection_reason",
    "selection_version",
    "synced_at_utc",
]


def rebuild_selected_markets(
    conn: sqlite3.Connection,
    *,
    min_volume: float = 20_000.0,
    force_remove: bool = False,
    selection_version: str = "v1_min_volume_only",
) -> dict[str, int]:
    logger.info(
        "Kalshi selected_markets rebuild started | min_volume=%s force_remove=%s selection_version=%s",
        min_volume,
        force_remove,
        selection_version,
    )
    source_df = pd.read_sql_query(
        """
        SELECT
            market_id,
            source,
            venue_market_id,
            event_id,
            venue_event_id,
            ticker,
            event_ticker,
            question,
            description,
            created_at,
            open_time,
            close_time,
            end_date,
            status,
            market_type,
            is_binary,
            is_resolved,
            is_active,
            is_closed,
            strike_type,
            custom_strike_json,
            volume_num,
            volume_24h_num,
            open_interest_num,
            liquidity_dollars,
            final_outcome,
            final_yes_probability,
            rules_primary,
            rules_secondary
        FROM raw_markets
        WHERE COALESCE(volume_num, 0) >= ?
        """,
        conn,
        params=[float(min_volume)],
    )
    if source_df.empty:
        if force_remove:
            conn.execute("DELETE FROM selected_markets")
        logger.info("Kalshi selected_markets rebuild finished | selected_rows=0")
        return {"selected_rows": 0}

    selected_df = source_df.copy()
    selected_df["series_ticker"] = None
    selected_df["event_title"] = None
    selected_df["kalshi_category"] = None
    selected_df["primary_domain"] = None
    selected_df["mutually_exclusive"] = None
    selected_df["selection_reason"] = f"volume_num>={float(min_volume):.0f}"
    selected_df["selection_version"] = selection_version
    selected_df["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
    for column in _SELECTED_MARKETS_COLUMNS:
        if column not in selected_df.columns:
            selected_df[column] = None
    selected_df = selected_df[_SELECTED_MARKETS_COLUMNS].copy()

    if force_remove:
        with conn:
            conn.execute("DELETE FROM selected_markets")
    written = upsert_selected_markets(conn, selected_df)
    logger.info("Kalshi selected_markets rebuild finished | selected_rows=%s", written)
    return {"selected_rows": int(written)}
