"""Registry refresh orchestration for metadata downloads."""

from __future__ import annotations

import json
import logging
import sqlite3

import pandas as pd

from clients.gamma_client import GammaClient
from collectors.markets_collector import MarketsCollector
from polymarket_registry.selection import filter_low_volume
from polymarket_registry.selection import filter_market_universe
from polymarket_registry.selection import enrich_candidates_with_tags
from polymarket_registry.upsert import replace_selected_markets
from polymarket_registry.upsert import upsert_market_universe


logger = logging.getLogger(__name__)
_MARKET_UNIVERSE_COLUMNS_LOGGED = False
_UNTAGGED_PRIMARY_DOMAIN = "unassigned"


def _to_log_sample_value(value: object) -> object:
    """Normalize one DataFrame cell into a compact JSON-safe log value."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (list, dict)):
        try:
            return json.loads(json.dumps(value, ensure_ascii=True, default=str))
        except Exception:
            return str(value)

    return value


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


def log_market_universe_columns_once(markets_df: pd.DataFrame) -> None:
    """Log fetched market-universe columns once per process for schema inspection."""
    global _MARKET_UNIVERSE_COLUMNS_LOGGED
    if _MARKET_UNIVERSE_COLUMNS_LOGGED:
        return

    columns = [str(column) for column in markets_df.columns]
    logger.info(
        "registry universe columns | count=%s columns=%s",
        len(columns),
        json.dumps(sorted(columns), ensure_ascii=True),
    )
    if not markets_df.empty:
        sample_row = {
            str(column): _to_log_sample_value(value)
            for column, value in markets_df.iloc[0].to_dict().items()
        }
        logger.info(
            "registry universe sample_row | row=%s",
            json.dumps(sample_row, ensure_ascii=True, default=str, sort_keys=True),
        )
    _MARKET_UNIVERSE_COLUMNS_LOGGED = True


def _with_empty_tag_metadata(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Attach empty tag metadata to a selected candidate dataframe."""
    work = candidate_df.copy()
    rows = len(work)
    work["tag_labels"] = [[] for _ in range(rows)]
    work["matched_tags"] = [[] for _ in range(rows)]
    work["matched_domains"] = [[] for _ in range(rows)]
    work["primary_domain"] = _UNTAGGED_PRIMARY_DOMAIN
    return work


def refresh_market_universe(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient,
    min_created_at: str,
    max_metadata_pages: int,
    page_limit: int = 200,
    include_active: bool = False,
    preserve_existing: bool = False,
) -> pd.DataFrame:
    """Refresh the broad raw market universe without research-specific filtering."""
    logger.info(
        "market universe refresh started | min_created_at=%s max_metadata_pages=%s page_limit=%s",
        min_created_at,
        max_metadata_pages,
        page_limit,
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
    existing_rows = conn.execute("SELECT COUNT(*) FROM market_universe").fetchone()[0]
    if preserve_existing:
        logger.info("market universe preserve_existing enabled | existing_rows=%s", existing_rows)
    else:
        conn.execute("DELETE FROM market_universe")
        logger.info("market universe cleared before refresh | deleted_rows=%s", existing_rows)
    logger.info("registry stage started | stage=download_market_universe")
    report = markets_collector.download_market_meta(
        include_active=include_active,
        include_closed=True,
        limit=page_limit,
        max_pages=max_metadata_pages,
        min_created_at=min_created_at,
        show_progress=True,
        estimate_total=False,
        frame_type="pandas",
    )
    markets_df = report["markets"]
    log_market_universe_columns_once(markets_df)
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
            u.event_description,
            u.event_start_time,
            u.event_score,
            u.event_period,
            u.event_series_id,
            u.event_recurrence,
            u.event_series_type,
            u.question,
            u.description,
            u.resolution_source,
            u.created_at,
            u.end_date,
            u.closed,
            u.archived,
            u.volume_num,
            u.liquidity_num,
            u.outcomes,
            u.outcome_prices,
            u.clob_token_ids,
            u.closed_time,
            u.uma_resolution_status,
            u.neg_risk,
            u.neg_risk_market_id,
            u.group_item_title,
            u.synced_at_utc
        FROM market_universe AS u
        WHERE u.created_at IS NOT NULL
          AND u.created_at >= ?
        ORDER BY u.volume_num DESC, u.created_at DESC
        """,
        conn,
        params=(min_created_at,),
    )


def select_market_registry_from_universe(
    conn: sqlite3.Connection,
    *,
    gamma: GammaClient | None,
    min_created_at: str,
    min_resolved_volume: float,
    tag_enrichment: bool = False,
) -> pd.DataFrame:
    """Build the filtered market registry from the local market_universe table."""
    logger.info(
        "registry selection started | source=market_universe min_created_at=%s tag_enrichment=%s",
        min_created_at,
        tag_enrichment,
    )
    logger.info("registry stage started | stage=load_market_universe")
    markets_df = load_market_universe_for_selection(conn, min_created_at=min_created_at)
    logger.info("registry stage finished | stage=load_market_universe loaded=%s", len(markets_df))
    resolved_probe = (
        filter_market_universe(
            markets_df,
            min_resolved_volume=0.0,
            apply_exclusion_blocks=False,
        )
        if not markets_df.empty
        else pd.DataFrame()
    )
    final_outcome_rows = len(resolved_probe)
    logger.info(
        "registry selection precondition | source_rows=%s source_resolved_candidate_rows=%s",
        len(markets_df),
        final_outcome_rows,
    )
    if final_outcome_rows <= 0:
        raise RuntimeError(
            "market_universe has no resolved Yes/No candidates from outcomes/outcome_prices; aborting market_selection before modifying selected_markets"
        )
    logger.info("registry stage started | stage=build_candidate_pool")
    raw_candidate_df = filter_market_universe(
        markets_df,
        min_resolved_volume=min_resolved_volume,
        apply_exclusion_blocks=True,
    )
    logger.info("registry stage finished | stage=build_candidate_pool candidates=%s", len(raw_candidate_df))
    logger.info(
        "registry selection ready | source_rows=%s resolved_candidates=%s",
        len(markets_df),
        len(raw_candidate_df),
    )
    if tag_enrichment:
        if gamma is None:
            raise RuntimeError("tag_enrichment=True requires a GammaClient")
        logger.info("registry stage started | stage=tag_enrichment candidates=%s", len(raw_candidate_df))
        enriched_df = enrich_candidates_with_tags(raw_candidate_df, gamma)
        logger.info("registry stage finished | stage=tag_enrichment enriched=%s", len(enriched_df))
        if enriched_df.empty:
            logger.warning("registry selection found no tag-enriched candidates")
            replace_selected_markets(conn, enriched_df)
            return enriched_df
        selected_df = enriched_df.copy()
        selected_df["primary_domain"] = selected_df["primary_domain"].fillna(_UNTAGGED_PRIMARY_DOMAIN)
        selected_counts = (
            selected_df["primary_domain"]
            .fillna(_UNTAGGED_PRIMARY_DOMAIN)
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
    else:
        logger.info("registry stage skipped | stage=tag_enrichment enabled=false")
        selected_df = _with_empty_tag_metadata(raw_candidate_df).reset_index(drop=True)
        selected_counts = {_UNTAGGED_PRIMARY_DOMAIN: len(selected_df)}
    logger.info(
        "registry stage started | stage=replace_filtered_markets selected=%s",
        len(selected_df),
    )
    replaced_rows = replace_selected_markets(conn, selected_df)
    logger.info(
        "registry stage finished | stage=replace_filtered_markets replaced_rows=%s",
        replaced_rows,
    )
    logger.info(
        "registry selection finished | selected_counts=%s replaced_rows=%s",
        json.dumps(selected_counts, ensure_ascii=True, sort_keys=True),
        replaced_rows,
    )
    return selected_df
