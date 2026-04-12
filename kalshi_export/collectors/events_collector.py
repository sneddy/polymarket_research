from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from clients.kalshi_client import KalshiClient
from collectors.markets_collector import _resolve_tqdm, _running_in_notebook, _to_bool_flag


logger = logging.getLogger(__name__)

_EVENT_METADATA_COLUMNS = [
    "event_id",
    "source",
    "venue_event_id",
    "event_ticker",
    "series_ticker",
    "event_title",
    "event_sub_title",
    "kalshi_category",
    "mutually_exclusive",
    "strike_period",
    "status",
    "created_at",
    "close_time",
    "last_updated_ts",
    "event_url",
    "rules_primary",
    "subtitle",
    "synced_at_utc",
]


def _normalize_event_row(event: dict[str, Any]) -> dict[str, Any]:
    event_ticker = str(event.get("event_ticker") or event.get("ticker") or "").strip()
    return {
        "event_id": f"kalshi:event:{event_ticker}" if event_ticker else None,
        "source": "kalshi",
        "venue_event_id": event_ticker or None,
        "event_ticker": event_ticker or None,
        "series_ticker": event.get("series_ticker"),
        "event_title": event.get("title"),
        "event_sub_title": event.get("sub_title"),
        "kalshi_category": event.get("category"),
        "mutually_exclusive": _to_bool_flag(event.get("mutually_exclusive")),
        "strike_period": event.get("strike_period"),
        "status": event.get("status"),
        "created_at": event.get("created_time"),
        "close_time": event.get("close_time"),
        "last_updated_ts": event.get("last_updated_ts"),
        "event_url": event.get("event_url"),
        "rules_primary": event.get("rules_primary"),
        "subtitle": event.get("sub_title"),
    }


class EventsCollector:
    def __init__(self, client: KalshiClient) -> None:
        self._client = client

    @staticmethod
    def prepare_event_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["synced_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
        for column in _EVENT_METADATA_COLUMNS:
            if column not in out.columns:
                out[column] = None
        return out[_EVENT_METADATA_COLUMNS].copy()

    def fetch_event_rows(
        self,
        *,
        event_tickers: list[str],
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        tqdm = _resolve_tqdm(show_progress=show_progress)
        pbar = None
        if show_progress and tqdm is not None:
            pbar = tqdm(
                total=len(event_tickers),
                disable=False,
                unit="event",
                desc="Enriching Kalshi events",
                leave=True,
            )
        elif show_progress and tqdm is None and not _running_in_notebook():
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        rows: list[dict[str, Any]] = []
        try:
            for idx, event_ticker in enumerate(event_tickers, start=1):
                payload = self._client.get_event(event_ticker, with_nested_markets=False)
                event_row = payload.get("event") if isinstance(payload, dict) else None
                if isinstance(event_row, dict):
                    rows.append(_normalize_event_row(event_row))
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"events": idx, "rows": len(rows)})
        finally:
            if pbar is not None:
                pbar.close()
        return rows
