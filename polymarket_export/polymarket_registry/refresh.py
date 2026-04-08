"""Registry refresh orchestration for metadata downloads."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence

import pandas as pd

from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector
from configs.resolved_dataset_domain_config import DOMAIN_PRIORITY
from polymarket_registry.selection import filter_low_volume
from polymarket_registry.selection import filter_market_universe
from polymarket_registry.selection import enrich_candidates_with_tags
from polymarket_registry.upsert import upsert_markets
from polymarket_registry.upsert import upsert_markets_for_categories
from polymarket_registry.upsert import replace_markets_for_categories
from polymarket_registry.upsert import upsert_market_universe


logger = logging.getLogger(__name__)


def log_market_universe_window(markets_df: pd.DataFrame) -> None:
    """Log the temporal window and edge examples of a fetched market universe."""
    if markets_df.empty:
        logger.info("registry universe window | fetched=0")
        return

    work = markets_df.copy()
    work["created_at_norm"] = pd.to_datetime(work.get("created_at"), utc=True, errors="coerce")
    valid = work.loc[work["created_at_norm"].notna()].sort_values("created_at_norm", ascending=True)

    if valid.empty:
        logger.info("registry universe window | fetched=%s created_at_valid=0", len(markets_df))
        return

    first_row = valid.iloc[0]
    last_row = valid.iloc[-1]
    logger.info(
        "registry universe window | fetched=%s min_created_at=%s min_market_id=%s min_slug=%s max_created_at=%s max_market_id=%s max_slug=%s",
        len(markets_df),
        first_row["created_at_norm"].isoformat(),
        first_row.get("id"),
        first_row.get("slug"),
        last_row["created_at_norm"].isoformat(),
        last_row.get("id"),
        last_row.get("slug"),
    )


def refresh_market_universe(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    min_created_at: str,
    max_metadata_pages: int,
    include_active: bool = False,
) -> pd.DataFrame:
    """Refresh the broad raw market universe without research-specific filtering."""
    logger.info(
        "market universe refresh started | min_created_at=%s max_metadata_pages=%s",
        min_created_at,
        max_metadata_pages,
    )
    logger.info(
        "market universe fetch mode | include_active=%s include_closed=%s note=%s",
        include_active,
        True,
        (
            "Universe refresh uses active + closed slices so min_created_at can reach historical markets"
            if include_active
            else "Universe refresh defaults to closed-only for research datasets; pass include_active=True for full universe"
        ),
    )
    markets_collector = MarketsCollector(gamma)
    logger.info("registry stage started | stage=download_market_universe")
    report = markets_collector.download_market_meta(
        include_active=include_active,
        include_closed=True,
        limit=200,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    log_market_universe_window(markets_df)
    logger.info("registry stage finished | stage=download_market_universe fetched=%s", len(markets_df))
    logger.info("registry stage started | stage=upsert_market_universe")
    universe_upserted = upsert_market_universe(conn, markets_df)
    logger.info("registry stage finished | stage=upsert_market_universe upserted=%s", universe_upserted)
    logger.info(
        "market universe refresh finished | fetched=%s universe_upserted=%s",
        len(markets_df),
        universe_upserted,
    )
    return markets_df


def refresh_market_registry(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    category: str,
    min_created_at: str,
    min_resolved_volume: float,
    max_metadata_pages: int,
) -> pd.DataFrame:
    """Refresh the filtered market registry for one category."""
    logger.info(
        "registry refresh started | category=%s min_created_at=%s max_metadata_pages=%s",
        category,
        min_created_at,
        max_metadata_pages,
    )
    logger.info(
        "registry refresh fetch mode | include_active=%s include_closed=%s note=%s",
        True,
        True,
        "Gamma /markets inactive filtering is unreliable, so the historical slice is fetched via closed=true",
    )
    markets_collector = MarketsCollector(gamma)
    logger.info("registry stage started | stage=download_market_universe")
    report = markets_collector.download_market_meta(
        include_active=True,
        include_closed=True,
        limit=200,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    log_market_universe_window(markets_df)
    logger.info("registry stage finished | stage=download_market_universe fetched=%s", len(markets_df))
    logger.info("registry stage started | stage=upsert_market_universe")
    universe_upserted = upsert_market_universe(conn, markets_df)
    logger.info("registry stage finished | stage=upsert_market_universe upserted=%s", universe_upserted)
    logger.info("registry stage started | stage=build_candidate_pool")
    raw_candidate_df = filter_low_volume(markets_df, min_resolved_volume=min_resolved_volume)
    logger.info("registry stage finished | stage=build_candidate_pool candidates=%s", len(raw_candidate_df))
    logger.info(
        "registry metadata ready | total_markets=%s universe_upserted=%s resolved_candidates=%s",
        len(markets_df),
        universe_upserted,
        len(raw_candidate_df),
    )
    logger.info("registry stage started | stage=tag_enrichment candidates=%s", len(raw_candidate_df))
    enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
    logger.info("registry stage finished | stage=tag_enrichment enriched=%s", len(enriched_df))
    if enriched_df.empty:
        logger.warning("registry refresh found no tag-enriched candidates | category=%s", category)
        return enriched_df

    selected_df = enriched_df.loc[enriched_df["primary_domain"] == category].reset_index(drop=True)
    logger.info(
        "registry stage started | stage=upsert_filtered_markets category=%s selected=%s",
        category,
        len(selected_df),
    )
    upserted = upsert_markets(conn, category, selected_df)
    logger.info(
        "registry stage finished | stage=upsert_filtered_markets category=%s upserted=%s",
        category,
        upserted,
    )
    logger.info(
        "registry refresh finished | category=%s selected=%s upserted=%s",
        category,
        len(selected_df),
        upserted,
    )
    return selected_df


def refresh_market_registry_all_categories(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    min_created_at: str,
    min_resolved_volume: float,
    max_metadata_pages: int,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Refresh the filtered market registry for all requested categories."""
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    logger.info(
        "registry refresh started | categories=%s min_created_at=%s max_metadata_pages=%s",
        ",".join(category_list),
        min_created_at,
        max_metadata_pages,
    )
    logger.info(
        "registry refresh fetch mode | include_active=%s include_closed=%s note=%s",
        True,
        True,
        "Gamma /markets inactive filtering is unreliable, so the historical slice is fetched via closed=true",
    )
    markets_collector = MarketsCollector(gamma)
    logger.info("registry stage started | stage=download_market_universe")
    report = markets_collector.download_market_meta(
        include_active=True,
        include_closed=True,
        limit=200,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    log_market_universe_window(markets_df)
    logger.info("registry stage finished | stage=download_market_universe fetched=%s", len(markets_df))
    logger.info("registry stage started | stage=upsert_market_universe")
    universe_upserted = upsert_market_universe(conn, markets_df)
    logger.info("registry stage finished | stage=upsert_market_universe upserted=%s", universe_upserted)
    logger.info("registry stage started | stage=build_candidate_pool")
    raw_candidate_df = filter_low_volume(markets_df, min_resolved_volume=min_resolved_volume)
    logger.info("registry stage finished | stage=build_candidate_pool candidates=%s", len(raw_candidate_df))
    logger.info(
        "registry metadata ready | total_markets=%s universe_upserted=%s resolved_candidates=%s",
        len(markets_df),
        universe_upserted,
        len(raw_candidate_df),
    )
    logger.info("registry stage started | stage=tag_enrichment candidates=%s", len(raw_candidate_df))
    enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
    logger.info("registry stage finished | stage=tag_enrichment enriched=%s", len(enriched_df))
    if enriched_df.empty:
        logger.warning("registry refresh found no tag-enriched candidates")
        return enriched_df

    selected_df = enriched_df.loc[enriched_df["primary_domain"].isin(category_list)].reset_index(drop=True)
    logger.info(
        "registry stage started | stage=upsert_filtered_markets categories=%s selected=%s",
        ",".join(category_list),
        len(selected_df),
    )
    upsert_counts = upsert_markets_for_categories(conn, selected_df, categories=category_list)
    logger.info(
        "registry stage finished | stage=upsert_filtered_markets upsert_counts=%s",
        json.dumps(upsert_counts, ensure_ascii=True, sort_keys=True),
    )
    selected_counts = {
        category: int((selected_df["primary_domain"] == category).sum())
        for category in category_list
    }
    logger.info(
        "registry refresh finished | selected_counts=%s upsert_counts=%s",
        json.dumps(selected_counts, ensure_ascii=True, sort_keys=True),
        json.dumps(upsert_counts, ensure_ascii=True, sort_keys=True),
    )
    return selected_df


def load_market_universe_for_selection(
    conn: sqlite3.Connection,
    *,
    min_created_at: str,
) -> pd.DataFrame:
    """Load the locally stored market universe used by market_selection."""
    return pd.read_sql_query(
        """
        SELECT
            u.market_id,
            u.condition_id,
            u.market_slug,
            u.event_id,
            u.event_slug,
            u.event_title,
            u.event_series_slug,
            u.question,
            u.description,
            u.resolution_source,
            u.created_at,
            u.end_date,
            u.active,
            u.closed,
            u.archived,
            u.volume_num,
            u.liquidity_num,
            u.final_outcome,
            u.synced_at_utc
        FROM market_universe AS u
        WHERE u.created_at IS NOT NULL
          AND u.created_at >= ?
        ORDER BY u.volume_num DESC, u.created_at DESC
        """,
        conn,
        params=(min_created_at,),
    )


def select_market_registry_from_universe_all_categories(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    min_created_at: str,
    min_resolved_volume: float,
    categories: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the filtered market registry from the local market_universe table."""
    category_list = list(categories) if categories is not None else list(DOMAIN_PRIORITY)
    logger.info(
        "registry selection started | source=market_universe categories=%s min_created_at=%s",
        ",".join(category_list),
        min_created_at,
    )
    logger.info("registry stage started | stage=load_market_universe")
    markets_df = load_market_universe_for_selection(conn, min_created_at=min_created_at)
    logger.info("registry stage finished | stage=load_market_universe loaded=%s", len(markets_df))
    final_outcome_rows = int(markets_df.get("final_outcome").notna().sum()) if not markets_df.empty else 0
    logger.info(
        "registry selection precondition | source_rows=%s source_final_outcome_rows=%s",
        len(markets_df),
        final_outcome_rows,
    )
    if final_outcome_rows <= 0:
        raise RuntimeError(
            "market_universe has no final_outcome values; aborting market_selection before modifying markets"
        )
    logger.info("registry stage started | stage=build_candidate_pool")
    raw_candidate_df = filter_market_universe(markets_df, min_resolved_volume=min_resolved_volume)
    logger.info("registry stage finished | stage=build_candidate_pool candidates=%s", len(raw_candidate_df))
    logger.info(
        "registry selection ready | source_rows=%s resolved_candidates=%s",
        len(markets_df),
        len(raw_candidate_df),
    )
    logger.info("registry stage started | stage=tag_enrichment candidates=%s", len(raw_candidate_df))
    enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
    logger.info("registry stage finished | stage=tag_enrichment enriched=%s", len(enriched_df))
    if enriched_df.empty:
        logger.warning("registry selection found no tag-enriched candidates")
        if category_list:
            placeholders = ",".join("?" for _ in category_list)
            conn.execute(
                f"DELETE FROM markets WHERE primary_domain IN ({placeholders})",
                tuple(category_list),
            )
        return enriched_df

    selected_df = enriched_df.loc[enriched_df["primary_domain"].isin(category_list)].reset_index(drop=True)
    logger.info(
        "registry stage started | stage=replace_filtered_markets categories=%s selected=%s",
        ",".join(category_list),
        len(selected_df),
    )
    upsert_counts = replace_markets_for_categories(conn, selected_df, categories=category_list)
    logger.info(
        "registry stage finished | stage=replace_filtered_markets upsert_counts=%s",
        json.dumps(upsert_counts, ensure_ascii=True, sort_keys=True),
    )
    selected_counts = {
        category: int((selected_df["primary_domain"] == category).sum())
        for category in category_list
    }
    logger.info(
        "registry selection finished | selected_counts=%s upsert_counts=%s",
        json.dumps(selected_counts, ensure_ascii=True, sort_keys=True),
        json.dumps(upsert_counts, ensure_ascii=True, sort_keys=True),
    )
    return selected_df
