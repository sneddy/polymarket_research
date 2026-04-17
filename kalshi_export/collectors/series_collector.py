from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import _resolve_tqdm


logger = logging.getLogger(__name__)

_RAW_SERIES_COLUMNS = [
    "series_ticker",
    "title",
    "subtitle",
    "category",
    "tags_json",
    "frequency",
    "status",
    "created_at",
    "updated_at",
    "close_time",
    "settlement_time",
    "raw_payload_json",
    "synced_at_utc",
]


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _normalize_series_row(series: dict[str, Any]) -> dict[str, Any]:
    ticker = str(series.get("ticker") or "").strip() or None
    return {
        "series_ticker": ticker,
        "title": series.get("title"),
        "subtitle": series.get("subtitle") or series.get("sub_title"),
        "category": series.get("category"),
        "tags_json": _json_text(series.get("tags")),
        "frequency": series.get("frequency"),
        "status": series.get("status"),
        "created_at": series.get("created_time"),
        "updated_at": series.get("updated_time") or series.get("last_updated_ts"),
        "close_time": series.get("close_time"),
        "settlement_time": series.get("settlement_time"),
        "raw_payload_json": _json_text(series),
    }


class SeriesCollector:
    def __init__(self, client: KalshiClient) -> None:
        self._client = client

    @staticmethod
    def prepare_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=_RAW_SERIES_COLUMNS)
        out = pd.concat(frames, ignore_index=True, sort=False)
        out = out.dropna(subset=["series_ticker"]).drop_duplicates(subset=["series_ticker"], keep="last").reset_index(drop=True)
        out["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
        for column in _RAW_SERIES_COLUMNS:
            if column not in out.columns:
                out[column] = None
        return out[_RAW_SERIES_COLUMNS].copy()

    def iter_series_batches(
        self,
        *,
        limit: int = 200,
        max_pages: int | None = None,
        show_progress: bool = True,
        **params: Any,
    ) -> list[list[dict[str, Any]]]:
        tqdm = _resolve_tqdm(show_progress=show_progress)
        pbar = (
            tqdm(total=max_pages, disable=False, unit="page", desc="Downloading Kalshi series", leave=True)
            if tqdm is not None
            else None
        )
        pages = 0
        rows_total = 0
        batches: list[list[dict[str, Any]]] = []
        try:
            batch: list[dict[str, Any]] = []
            for row in self._client.iter_series(limit=limit, max_pages=max_pages, **params):
                if isinstance(row, dict):
                    batch.append(_normalize_series_row(row))
                if len(batch) >= int(limit):
                    batches.append(batch)
                    pages += 1
                    rows_total += len(batch)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"pages": pages, "rows": rows_total})
                    batch = []
            if batch:
                batches.append(batch)
                pages += 1
                rows_total += len(batch)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"pages": pages, "rows": rows_total})
        finally:
            if pbar is not None:
                pbar.close()
        return batches
