from __future__ import annotations

import logging
import sqlite3
import time
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import MarketsCollector, _resolve_tqdm
from collectors.series_collector import SeriesCollector
from kalshi_registry.upsert import upsert_raw_markets, upsert_raw_series


logger = logging.getLogger(__name__)
MarketEndpoint = Literal["markets", "historical"]


def _format_unix_ts_utc(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value), tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _is_non_mve_market(row: dict[str, Any]) -> bool:
    collection = str(row.get("mve_collection_ticker") or "").strip()
    selected_legs = str(row.get("mve_selected_legs_json") or "").strip()
    return not collection and selected_legs in {"", "[]", "null"}


def refresh_raw_series(
    conn: sqlite3.Connection,
    *,
    kalshi: KalshiClient,
    limit: int = 200,
    max_pages: int | None = None,
    force_remove: bool = False,
) -> dict[str, int]:
    logger.info(
        "Kalshi raw series refresh started | limit=%s max_pages=%s force_remove=%s",
        limit,
        max_pages,
        force_remove,
    )
    existing_rows = conn.execute("SELECT COUNT(*) FROM raw_series").fetchone()[0]
    if force_remove:
        conn.execute("DELETE FROM raw_series")
        logger.info("Kalshi raw_series cleared before refresh | deleted_rows=%s", existing_rows)
    else:
        logger.info("Kalshi raw_series preserving existing rows by default | existing_rows=%s", existing_rows)

    collector = SeriesCollector(kalshi)
    frames = [pd.DataFrame(batch) for batch in collector.iter_series_batches(limit=limit, max_pages=max_pages, show_progress=True)]
    series_df = collector.prepare_frame(frames)
    written = upsert_raw_series(conn, series_df)
    table_rows = conn.execute("SELECT COUNT(*) FROM raw_series").fetchone()[0]
    logger.info("Kalshi raw series refresh finished | fetched_rows=%s table_rows=%s", written, table_rows)
    return {"fetched_rows": int(written), "table_rows": int(table_rows)}


def refresh_raw_markets_from_selected_series(
    conn: sqlite3.Connection,
    *,
    kalshi: KalshiClient,
    endpoint: MarketEndpoint = "markets",
    limit: int = 200,
    max_pages_per_series: int | None = None,
    force_remove: bool = False,
    write_batch_pages: int = 100,
    status: str | None = None,
    min_close_ts: int | None = None,
    max_close_ts: int | None = None,
    exclude_mve: bool = True,
) -> dict[str, int]:
    logger.info(
        "Kalshi raw markets refresh started | endpoint=%s limit=%s max_pages_per_series=%s force_remove=%s write_batch_pages=%s status=%s min_close_date=%s max_close_date=%s exclude_mve=%s series_pause_seconds=%s",
        endpoint,
        limit,
        max_pages_per_series,
        force_remove,
        write_batch_pages,
        status,
        _format_unix_ts_utc(min_close_ts),
        _format_unix_ts_utc(max_close_ts),
        exclude_mve,
        kalshi._http.series_pause_seconds,
    )
    existing_rows = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
    if force_remove:
        conn.execute("DELETE FROM raw_markets")
        logger.info("Kalshi raw_markets cleared before refresh | deleted_rows=%s", existing_rows)
    else:
        logger.info("Kalshi raw_markets preserving existing rows by default | existing_rows=%s", existing_rows)

    selected_series = pd.read_sql_query(
        "SELECT series_ticker FROM selected_series WHERE series_ticker IS NOT NULL ORDER BY series_ticker",
        conn,
    )
    selected_series_tickers = [str(value).strip() for value in selected_series["series_ticker"].tolist() if str(value).strip()]
    if not selected_series_tickers:
        logger.info("Kalshi raw markets refresh finished | selected_series_rows=0")
        return {
            "fetched_rows": 0,
            "table_rows": int(existing_rows),
            "selected_series_rows": 0,
            "remaining_series_rows": 0,
        }
    series_tickers = selected_series_tickers
    logger.info(
        "Kalshi raw markets selected_series coverage before refresh | selected_series_rows=%s remaining_series_rows=%s",
        len(selected_series_tickers),
        len(series_tickers),
    )
    if not series_tickers:
        table_rows = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
        logger.info(
            "Kalshi raw markets refresh finished | nothing_to_download selected_series_rows=%s remaining_series_rows=0",
            len(selected_series_tickers),
        )
        return {
            "fetched_rows": 0,
            "table_rows": int(table_rows),
            "selected_series_rows": int(len(selected_series_tickers)),
            "remaining_series_rows": 0,
        }

    tqdm = _resolve_tqdm(show_progress=True)
    pbar = (
        tqdm(total=len(series_tickers), disable=False, unit="series", desc="Downloading Kalshi raw markets", leave=True)
        if tqdm is not None
        else None
    )

    collector = MarketsCollector(kalshi)
    pending_frames: list[pd.DataFrame] = []
    fetched_rows = 0
    flushed_batches = 0
    attempted_series = 0
    completed_series = 0
    page_counter = 0

    def _flush_pending() -> int:
        nonlocal pending_frames, flushed_batches
        if not pending_frames:
            return 0
        batch_df = collector._prepare_index_frame(pending_frames)
        if batch_df.empty:
            pending_frames = []
            return 0
        written = upsert_raw_markets(conn, batch_df)
        flushed_batches += 1
        logger.info("Kalshi raw_markets batch upserted | batch_index=%s rows=%s", flushed_batches, written)
        pending_frames = []
        return written

    iter_params: dict[str, Any] = {}
    if endpoint == "markets" and status:
        iter_params["status"] = status
    if endpoint == "markets" and min_close_ts is not None:
        iter_params["min_close_ts"] = int(min_close_ts)
    if endpoint == "markets" and max_close_ts is not None:
        iter_params["max_close_ts"] = int(max_close_ts)
    if endpoint == "markets" and exclude_mve:
        iter_params["mve_filter"] = "exclude"

    try:
        for idx, series_ticker in enumerate(series_tickers, start=1):
            attempted_series += 1
            series_rows = 0
            for batch_rows in collector.iter_market_index_batches(
                endpoint=endpoint,
                limit=limit,
                max_pages=max_pages_per_series,
                show_progress=False,
                estimate_total=False,
                series_ticker=series_ticker,
                **iter_params,
            ):
                if endpoint == "historical" and exclude_mve:
                    batch_rows = [row for row in batch_rows if _is_non_mve_market(row)]
                for row in batch_rows:
                    if not row.get("series_ticker"):
                        row["series_ticker"] = series_ticker
                if batch_rows:
                    pending_frames.append(pd.DataFrame(batch_rows))
                    fetched_rows += len(batch_rows)
                    series_rows += len(batch_rows)
                page_counter += 1
                if page_counter % max(1, int(write_batch_pages)) == 0:
                    _flush_pending()
            if series_rows > 0:
                _flush_pending()
            completed_series += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"series": idx, "rows": fetched_rows, "last_series_rows": series_rows, "completed": completed_series})
            if kalshi._http.series_pause_seconds > 0 and idx < len(series_tickers):
                time.sleep(kalshi._http.series_pause_seconds)
    finally:
        if pbar is not None:
            pbar.close()

    _flush_pending()
    table_rows = conn.execute("SELECT COUNT(*) FROM raw_markets").fetchone()[0]
    remaining_series_rows = len(selected_series_tickers) - completed_series
    logger.info(
        "Kalshi raw markets refresh finished | fetched_rows=%s table_rows=%s selected_series_rows=%s attempted_series_rows=%s completed_series_rows=%s remaining_series_rows=%s flushed_batches=%s",
        fetched_rows,
        table_rows,
        len(selected_series_tickers),
        attempted_series,
        completed_series,
        remaining_series_rows,
        flushed_batches,
    )
    return {
        "fetched_rows": int(fetched_rows),
        "table_rows": int(table_rows),
        "selected_series_rows": int(len(selected_series_tickers)),
        "attempted_series_rows": int(attempted_series),
        "completed_series_rows": int(completed_series),
        "remaining_series_rows": int(remaining_series_rows),
        "flushed_batches": int(flushed_batches),
    }
