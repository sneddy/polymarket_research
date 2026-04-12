"""Registry refresh orchestration for Kalshi metadata downloads."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import MarketsCollector
from kalshi_registry.upsert import upsert_raw_markets


logger = logging.getLogger(__name__)


def _format_unix_ts_utc(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _to_log_sample_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def refresh_raw_markets(
    conn: sqlite3.Connection,
    *,
    kalshi: KalshiClient,
    historical: bool = False,
    limit: int = 200,
    max_pages: int | None = None,
    force_remove: bool = False,
    write_batch_pages: int = 100,
    status: str | None = None,
    series_ticker: str | None = None,
    event_ticker: str | None = None,
    min_close_ts: int | None = None,
    max_close_ts: int | None = None,
) -> dict[str, int]:
    """Refresh the Kalshi raw markets table from `/markets` or `/historical/markets`."""
    endpoint = "historical/markets" if historical else "markets"
    if historical and status is not None:
        raise ValueError("`--status` is only supported for live `/markets` indexing.")
    if historical and series_ticker is not None:
        raise ValueError("`--series-ticker` is only supported for live `/markets` indexing.")
    if historical and (min_close_ts is not None or max_close_ts is not None):
        raise ValueError("`--min-close-date` and `--max-close-date` are only supported for live `/markets` indexing.")
    logger.info(
        "Kalshi raw markets refresh started | endpoint=%s limit=%s max_pages=%s force_remove=%s write_batch_pages=%s status=%s series_ticker=%s event_ticker=%s min_close_date=%s max_close_date=%s",
        endpoint,
        limit,
        max_pages,
        force_remove,
        write_batch_pages,
        status,
        series_ticker,
        event_ticker,
        _format_unix_ts_utc(min_close_ts),
        _format_unix_ts_utc(max_close_ts),
    )
    existing_rows = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
    if force_remove:
        conn.execute("DELETE FROM raw_markets")
        logger.info("Kalshi raw_markets cleared before refresh | deleted_rows=%s", existing_rows)
    else:
        logger.info("Kalshi raw_markets preserving existing rows by default | existing_rows=%s", existing_rows)

    markets_collector = MarketsCollector(kalshi)
    batch_pages = max(1, int(write_batch_pages))
    pending_frames: list[pd.DataFrame] = []
    fetched_rows = 0
    flushed_batches = 0
    sample_logged = False

    def _flush_pending() -> int:
        nonlocal pending_frames, flushed_batches, sample_logged
        if not pending_frames:
            return 0
        batch_df = markets_collector._prepare_index_frame(pending_frames)
        if batch_df.empty:
            pending_frames = []
            return 0
        if not sample_logged:
            logger.info(
                "Kalshi raw_markets columns | count=%s columns=%s",
                len(batch_df.columns),
                str(sorted([str(column) for column in batch_df.columns])),
            )
            sample_row = {
                str(column): _to_log_sample_value(value)
                for column, value in batch_df.iloc[0].to_dict().items()
            }
            logger.info("Kalshi raw_markets sample_row | row=%s", sample_row)
            sample_logged = True
        written = upsert_raw_markets(conn, batch_df)
        flushed_batches += 1
        logger.info(
            "Kalshi raw_markets batch upserted | batch_index=%s rows=%s",
            flushed_batches,
            written,
        )
        pending_frames = []
        return written

    page_counter = 0
    iter_params: dict[str, Any] = {}
    if status:
        iter_params["status"] = status
    if series_ticker:
        iter_params["series_ticker"] = series_ticker
    if event_ticker:
        iter_params["event_ticker"] = event_ticker
    if min_close_ts is not None:
        iter_params["min_close_ts"] = int(min_close_ts)
    if max_close_ts is not None:
        iter_params["max_close_ts"] = int(max_close_ts)

    for batch_rows in markets_collector.iter_market_index_batches(
        endpoint="historical" if historical else "markets",
        limit=limit,
        max_pages=max_pages,
        show_progress=True,
        estimate_total=True,
        **iter_params,
    ):
        if batch_rows:
            pending_frames.append(pd.DataFrame(batch_rows))
            fetched_rows += len(batch_rows)
        page_counter += 1
        if page_counter % batch_pages == 0:
            _flush_pending()
    _flush_pending()

    table_rows = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
    logger.info(
        "Kalshi raw_markets refresh finished | fetched_rows=%s table_rows=%s flushed_batches=%s",
        fetched_rows,
        table_rows,
        flushed_batches,
    )
    return {
        "fetched_rows": int(fetched_rows),
        "table_rows": int(table_rows),
        "flushed_batches": int(flushed_batches),
    }
