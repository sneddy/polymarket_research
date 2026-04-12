from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.events_collector import EventsCollector
from kalshi_registry.upsert import upsert_event_metadata, upsert_market_universe, upsert_selected_markets


logger = logging.getLogger(__name__)


def enrich_selected_markets(
    conn: sqlite3.Connection,
    *,
    kalshi: KalshiClient,
    refresh_existing: bool = False,
) -> dict[str, int]:
    logger.info("Kalshi enrichment started | refresh_existing=%s", refresh_existing)
    if refresh_existing:
        query = """
        SELECT DISTINCT sm.event_ticker
        FROM selected_markets sm
        WHERE sm.event_id IS NOT NULL
          AND sm.event_ticker IS NOT NULL
        """
    else:
        query = """
        SELECT DISTINCT sm.event_ticker
        FROM selected_markets sm
        LEFT JOIN event_metadata em ON sm.event_id = em.event_id
        WHERE sm.event_id IS NOT NULL
          AND sm.event_ticker IS NOT NULL
          AND em.event_id IS NULL
        """
    pending_events = pd.read_sql_query(query, conn)
    event_tickers = [str(value).strip() for value in pending_events["event_ticker"].dropna().tolist() if str(value).strip()]

    events_collector = EventsCollector(kalshi)
    event_rows = events_collector.fetch_event_rows(event_tickers=event_tickers, show_progress=True) if event_tickers else []
    events_df = events_collector.prepare_event_frame(event_rows)
    upserted_event_rows = upsert_event_metadata(conn, events_df) if not events_df.empty else 0

    enriched_df = pd.read_sql_query(
        """
        SELECT
            sm.market_id,
            sm.source,
            sm.venue_market_id,
            sm.event_id,
            sm.venue_event_id,
            COALESCE(em.series_ticker, sm.series_ticker) AS series_ticker,
            sm.ticker,
            sm.question,
            COALESCE(sm.description, em.subtitle, em.rules_primary) AS description,
            em.event_title AS event_title,
            em.kalshi_category AS kalshi_category,
            sm.primary_domain,
            sm.created_at,
            sm.open_time,
            sm.close_time,
            sm.end_date,
            sm.status,
            sm.market_type,
            sm.is_binary,
            sm.is_resolved,
            sm.is_active,
            sm.is_closed,
            em.mutually_exclusive AS mutually_exclusive,
            sm.strike_type,
            sm.custom_strike_json,
            sm.volume_num,
            sm.volume_24h_num,
            sm.open_interest_num,
            sm.liquidity_dollars,
            sm.final_outcome,
            sm.final_yes_probability,
            sm.rules_primary,
            sm.rules_secondary,
            sm.selection_reason,
            sm.selection_version,
            CURRENT_TIMESTAMP AS synced_at_utc
        FROM selected_markets sm
        LEFT JOIN event_metadata em ON sm.event_id = em.event_id
        """,
        conn,
    )

    selected_update_df = enriched_df.copy()
    if "synced_at_utc" in selected_update_df.columns:
        selected_update_df["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated_selected_rows = upsert_selected_markets(conn, selected_update_df)

    universe_df = pd.read_sql_query(
        """
        SELECT
            mi.market_id,
            mi.source,
            mi.venue_market_id,
            mi.event_id,
            mi.venue_event_id,
            em.series_ticker AS series_ticker,
            mi.ticker,
            mi.event_ticker,
            mi.title,
            mi.question,
            mi.subtitle,
            mi.yes_sub_title,
            mi.no_sub_title,
            mi.market_type,
            mi.status,
            em.event_title AS event_title,
            em.event_sub_title AS event_sub_title,
            em.kalshi_category AS kalshi_category,
            em.mutually_exclusive AS mutually_exclusive,
            em.strike_period AS strike_period,
            mi.rules_primary,
            mi.rules_secondary,
            mi.can_close_early,
            mi.early_close_condition,
            mi.is_provisional,
            mi.result,
            mi.settlement_value_dollars,
            mi.created_at,
            mi.updated_at,
            mi.open_time,
            mi.close_time,
            mi.expected_expiration_time,
            mi.expiration_time,
            mi.latest_expiration_time,
            mi.settlement_ts,
            mi.last_price_dollars,
            mi.previous_price_dollars,
            mi.yes_bid_dollars,
            mi.yes_ask_dollars,
            mi.no_bid_dollars,
            mi.no_ask_dollars,
            mi.yes_bid_size_fp,
            mi.yes_ask_size_fp,
            mi.volume_num,
            mi.volume_24h_num,
            mi.open_interest_num,
            mi.liquidity_dollars,
            mi.notional_value_dollars,
            mi.response_price_units,
            mi.price_level_structure,
            mi.tick_size,
            mi.strike_type,
            mi.floor_strike,
            mi.cap_strike,
            mi.functional_strike,
            mi.custom_strike_json,
            mi.mve_collection_ticker,
            mi.mve_selected_legs_json,
            mi.description,
            mi.end_date,
            mi.final_outcome,
            mi.final_yes_probability,
            mi.is_binary,
            mi.is_resolved,
            mi.is_active,
            mi.is_closed,
            mi.data_source_kind,
            CURRENT_TIMESTAMP AS synced_at_utc
        FROM raw_markets mi
        INNER JOIN selected_markets sm ON sm.market_id = mi.market_id
        LEFT JOIN event_metadata em ON mi.event_id = em.event_id
        """,
        conn,
    )
    if not universe_df.empty:
        universe_df["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
    upserted_universe_rows = upsert_market_universe(conn, universe_df) if not universe_df.empty else 0

    logger.info(
        "Kalshi enrichment finished | fetched_events=%s upserted_event_rows=%s updated_selected_rows=%s upserted_universe_rows=%s",
        len(event_tickers),
        upserted_event_rows,
        updated_selected_rows,
        upserted_universe_rows,
    )
    return {
        "fetched_events": int(len(event_tickers)),
        "upserted_event_rows": int(upserted_event_rows),
        "updated_selected_rows": int(updated_selected_rows),
        "upserted_universe_rows": int(upserted_universe_rows),
    }
