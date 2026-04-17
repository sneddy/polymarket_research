from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import _resolve_tqdm
from kalshi_registry.upsert import upsert_added_markets, upsert_minute_candles, upsert_probabilities


logger = logging.getLogger(__name__)

_MINUTE_CANDLE_COLUMNS = [
    "market_id",
    "source",
    "venue_market_id",
    "timestamp_utc",
    "yes_open_probability",
    "yes_high_probability",
    "yes_low_probability",
    "yes_close_probability",
    "yes_mean_probability",
    "volume_num",
    "open_interest_num",
]

_PROBABILITY_COLUMNS = [
    "market_id",
    "timestamp_utc",
    "yes_probability",
    "observed_trade",
    "trade_count",
    "total_size",
    "last_trade_price",
]

_ADDED_MARKETS_COLUMNS = [
    "market_id",
    "source",
    "venue_market_id",
    "series_ticker",
    "primary_domain",
    "added_at_utc",
    "storage_path",
    "history_source_mode",
    "probability_rows",
    "probability_start_utc",
    "probability_end_utc",
    "candle_rows_1m",
    "raw_trade_rows",
    "raw_trade_start_utc",
    "raw_trade_end_utc",
    "raw_trades_saved",
    "cutoff_ts_used",
    "download_warnings_json",
]


def _to_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _to_unix_seconds(value: object) -> int | None:
    ts = _to_timestamp(value)
    if ts is None:
        return None
    return int(ts.timestamp())


def _to_iso_utc_from_unix(value: object) -> str | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(int(value), unit="s", tz="UTC")
    except Exception:
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def build_pending_queue(conn: sqlite3.Connection, *, force_refresh: bool = False) -> pd.DataFrame:
    query = """
    SELECT
        sm.market_id,
        sm.source,
        sm.venue_market_id,
        sm.series_ticker,
        sm.ticker,
        sm.primary_domain,
        sm.history_start_utc,
        sm.history_end_utc,
        sm.settlement_ts,
        sm.status
    FROM selected_markets sm
    LEFT JOIN added_markets am ON sm.market_id = am.market_id
    WHERE COALESCE(sm.history_ready, 0) = 1
      AND (? = 1 OR am.market_id IS NULL)
    ORDER BY COALESCE(sm.history_end_utc, sm.close_time, sm.created_at), sm.market_id
    """
    return pd.read_sql_query(query, conn, params=[1 if force_refresh else 0])


def get_market_settled_cutoff(kalshi: KalshiClient) -> pd.Timestamp:
    payload = kalshi.get_historical_cutoff()
    value = payload.get("market_settled_ts") if isinstance(payload, dict) else None
    ts = _to_timestamp(value)
    if ts is None:
        raise RuntimeError("Kalshi historical cutoff response did not include a valid `market_settled_ts`.")
    return ts


def decide_history_source(market_row: pd.Series, cutoff_ts: pd.Timestamp) -> str:
    settlement_ts = _to_timestamp(market_row.get("settlement_ts"))
    if settlement_ts is not None and settlement_ts < cutoff_ts:
        return "historical"
    return "live"


def _iter_time_chunks(start_ts: int, end_ts: int, *, chunk_days: int) -> list[tuple[int, int]]:
    if end_ts < start_ts:
        return []
    chunk_seconds = max(1, int(chunk_days)) * 24 * 60 * 60
    chunks: list[tuple[int, int]] = []
    current_start = int(start_ts)
    final_end = int(end_ts)
    while current_start <= final_end:
        current_end = min(final_end, current_start + chunk_seconds - 1)
        chunks.append((current_start, current_end))
        current_start = current_end + 1
    return chunks


def _normalize_candles(
    *,
    market_row: pd.Series,
    candlesticks: list[dict[str, Any]],
    source_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live = source_mode == "live"
    for candle in candlesticks:
        if not isinstance(candle, dict):
            continue
        price = candle.get("price") if isinstance(candle.get("price"), dict) else {}
        rows.append(
            {
                "market_id": market_row["market_id"],
                "source": market_row["source"],
                "venue_market_id": market_row["venue_market_id"],
                "timestamp_utc": _to_iso_utc_from_unix(candle.get("end_period_ts")),
                "yes_open_probability": _to_float(price.get("open_dollars" if live else "open")),
                "yes_high_probability": _to_float(price.get("high_dollars" if live else "high")),
                "yes_low_probability": _to_float(price.get("low_dollars" if live else "low")),
                "yes_close_probability": _to_float(price.get("close_dollars" if live else "close")),
                "yes_mean_probability": _to_float(price.get("mean_dollars" if live else "mean")),
                "volume_num": _to_float(candle.get("volume_fp" if live else "volume")),
                "open_interest_num": _to_float(candle.get("open_interest_fp" if live else "open_interest")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        for column in _MINUTE_CANDLE_COLUMNS:
            out[column] = []
        return out
    out = out.dropna(subset=["timestamp_utc"]).copy()
    out = out.sort_values("timestamp_utc", kind="stable").drop_duplicates(subset=["market_id", "timestamp_utc"], keep="last")
    return out[_MINUTE_CANDLE_COLUMNS].copy()


def _download_candles_for_source(
    kalshi: KalshiClient,
    *,
    market_row: pd.Series,
    source_mode: str,
    chunk_days: int,
    show_progress: bool,
    progress_desc: str,
) -> pd.DataFrame:
    start_ts = _to_unix_seconds(market_row.get("history_start_utc"))
    end_ts = _to_unix_seconds(market_row.get("history_end_utc"))
    if start_ts is None or end_ts is None:
        return pd.DataFrame(columns=_MINUTE_CANDLE_COLUMNS)
    end_ts = min(end_ts, int(datetime.now(tz=UTC).timestamp()))
    if end_ts < start_ts:
        return pd.DataFrame(columns=_MINUTE_CANDLE_COLUMNS)

    chunks = _iter_time_chunks(start_ts, end_ts, chunk_days=chunk_days)
    tqdm = _resolve_tqdm(show_progress=show_progress)
    pbar = (
        tqdm(total=len(chunks), disable=False, unit="chunk", desc=progress_desc, leave=False)
        if tqdm is not None and len(chunks) > 1
        else None
    )

    frames: list[pd.DataFrame] = []
    try:
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            if source_mode == "historical":
                payload = kalshi.get_historical_market_candlesticks(
                    ticker=str(market_row["ticker"]),
                    start_ts=chunk_start,
                    end_ts=chunk_end,
                    period_interval=1,
                )
            else:
                payload = kalshi.get_market_candlesticks(
                    series_ticker=str(market_row["series_ticker"]),
                    ticker=str(market_row["ticker"]),
                    start_ts=chunk_start,
                    end_ts=chunk_end,
                    period_interval=1,
                )
            candlesticks = payload.get("candlesticks") if isinstance(payload, dict) else []
            batch_df = _normalize_candles(
                market_row=market_row,
                candlesticks=candlesticks if isinstance(candlesticks, list) else [],
                source_mode=source_mode,
            )
            if not batch_df.empty:
                frames.append(batch_df)
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"rows": sum(len(frame) for frame in frames), "chunk": chunk_idx})
    finally:
        if pbar is not None:
            pbar.close()

    if not frames:
        return pd.DataFrame(columns=_MINUTE_CANDLE_COLUMNS)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values("timestamp_utc", kind="stable").drop_duplicates(subset=["market_id", "timestamp_utc"], keep="last")
    return out[_MINUTE_CANDLE_COLUMNS].copy()


def download_market_minute_candles(
    kalshi: KalshiClient,
    *,
    market_row: pd.Series,
    cutoff_ts: pd.Timestamp,
    chunk_days: int,
    show_progress: bool,
    progress_desc: str,
) -> tuple[pd.DataFrame, str, list[str]]:
    warnings: list[str] = []
    primary = decide_history_source(market_row, cutoff_ts)
    alternate = "historical" if primary == "live" else "live"

    primary_error: Exception | None = None
    try:
        minute_df = _download_candles_for_source(
            kalshi,
            market_row=market_row,
            source_mode=primary,
            chunk_days=chunk_days,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        if not minute_df.empty:
            return minute_df, f"candles_{primary}", warnings
        warnings.append(f"primary_source_empty:{primary}")
    except Exception as exc:  # noqa: BLE001
        primary_error = exc
        warnings.append(f"primary_source_error:{primary}:{type(exc).__name__}")

    try:
        fallback_df = _download_candles_for_source(
            kalshi,
            market_row=market_row,
            source_mode=alternate,
            chunk_days=chunk_days,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        if not fallback_df.empty:
            warnings.append(f"fallback_source_used:{alternate}")
            return fallback_df, f"candles_{alternate}_fallback", warnings
        warnings.append(f"fallback_source_empty:{alternate}")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"fallback_source_error:{alternate}:{type(exc).__name__}")
        if primary_error is not None:
            raise RuntimeError(
                f"Both candle sources failed for market_id={market_row['market_id']}: "
                f"primary={type(primary_error).__name__}, fallback={type(exc).__name__}"
            ) from exc

    if primary_error is not None:
        raise primary_error
    return pd.DataFrame(columns=_MINUTE_CANDLE_COLUMNS), f"candles_{primary}", warnings


def build_probability_series_5m_from_candles(minute_df: pd.DataFrame, market_id: str) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS)

    work = minute_df.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc", kind="stable")
    if work.empty:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS)

    work = work.set_index("timestamp_utc")
    close_5m = work["yes_close_probability"].resample("5min", label="right", closed="right").last()
    close_5m = close_5m.ffill().dropna()
    if close_5m.empty:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS)

    observed_5m = (
        work["yes_close_probability"]
        .resample("5min", label="right", closed="right")
        .count()
        .reindex(close_5m.index, fill_value=0)
        .gt(0)
        .astype(int)
    )
    total_size_5m = (
        work["volume_num"]
        .fillna(0.0)
        .resample("5min", label="right", closed="right")
        .sum()
        .reindex(close_5m.index, fill_value=0.0)
    )

    out = pd.DataFrame(
        {
            "market_id": market_id,
            "timestamp_utc": close_5m.index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "yes_probability": close_5m.astype(float).values,
            "observed_trade": observed_5m.astype(int).values,
            "trade_count": 0,
            "total_size": total_size_5m.astype(float).values,
            "last_trade_price": close_5m.astype(float).values,
        }
    )
    return out[_PROBABILITY_COLUMNS].copy()


def store_market_history(
    conn: sqlite3.Connection,
    *,
    market_row: pd.Series,
    minute_df: pd.DataFrame,
    probability_df: pd.DataFrame,
    storage_path: str,
    history_source_mode: str,
    cutoff_ts_used: str,
    warnings: list[str],
) -> dict[str, Any]:
    market_id = str(market_row["market_id"])
    with conn:
        conn.execute("DELETE FROM minute_candles WHERE market_id = ?", (market_id,))
        conn.execute("DELETE FROM probabilities WHERE market_id = ?", (market_id,))

    minute_rows = upsert_minute_candles(conn, minute_df[_MINUTE_CANDLE_COLUMNS].copy()) if not minute_df.empty else 0
    probability_rows = upsert_probabilities(conn, probability_df[_PROBABILITY_COLUMNS].copy()) if not probability_df.empty else 0

    added_at_utc = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
    added_df = pd.DataFrame(
        [
            {
                "market_id": market_id,
                "source": market_row.get("source") or "kalshi",
                "venue_market_id": market_row.get("venue_market_id") or market_row.get("ticker"),
                "series_ticker": market_row.get("series_ticker"),
                "primary_domain": market_row.get("primary_domain"),
                "added_at_utc": added_at_utc,
                "storage_path": storage_path,
                "history_source_mode": history_source_mode,
                "probability_rows": int(probability_rows),
                "probability_start_utc": None if probability_df.empty else str(probability_df["timestamp_utc"].min()),
                "probability_end_utc": None if probability_df.empty else str(probability_df["timestamp_utc"].max()),
                "candle_rows_1m": int(minute_rows),
                "raw_trade_rows": None,
                "raw_trade_start_utc": None,
                "raw_trade_end_utc": None,
                "raw_trades_saved": 0,
                "cutoff_ts_used": cutoff_ts_used,
                "download_warnings_json": json.dumps(warnings, sort_keys=True),
            }
        ]
    )
    upsert_added_markets(conn, added_df[_ADDED_MARKETS_COLUMNS].copy())
    return {
        "minute_rows": int(minute_rows),
        "probability_rows": int(probability_rows),
    }
