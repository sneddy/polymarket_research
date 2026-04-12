from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import Any
from typing import Literal

import pandas as pd

from clients.kalshi_client import KalshiClient


logger = logging.getLogger(__name__)

FrameType = Literal["pandas"]
IndexEndpoint = Literal["markets", "historical"]
_RAW_MARKETS_COLUMNS = [
    "market_id",
    "source",
    "venue_market_id",
    "event_id",
    "venue_event_id",
    "ticker",
    "event_ticker",
    "title",
    "question",
    "subtitle",
    "yes_sub_title",
    "no_sub_title",
    "market_type",
    "status",
    "created_at",
    "updated_at",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "expiration_time",
    "latest_expiration_time",
    "settlement_ts",
    "last_price_dollars",
    "previous_price_dollars",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "yes_bid_size_fp",
    "yes_ask_size_fp",
    "volume_num",
    "volume_24h_num",
    "open_interest_num",
    "liquidity_dollars",
    "notional_value_dollars",
    "response_price_units",
    "price_level_structure",
    "tick_size",
    "strike_type",
    "floor_strike",
    "cap_strike",
    "functional_strike",
    "custom_strike_json",
    "mve_collection_ticker",
    "mve_selected_legs_json",
    "rules_primary",
    "rules_secondary",
    "can_close_early",
    "early_close_condition",
    "is_provisional",
    "result",
    "settlement_value_dollars",
    "description",
    "end_date",
    "final_outcome",
    "final_yes_probability",
    "is_binary",
    "is_resolved",
    "is_active",
    "is_closed",
    "data_source_kind",
    "indexed_at_utc",
]


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _resolve_tqdm(show_progress: bool) -> Any | None:
    if not show_progress:
        return None

    if _running_in_notebook():
        try:
            from tqdm.notebook import tqdm as _tqdm

            return _tqdm
        except Exception:
            pass

    try:
        from tqdm.auto import tqdm as _tqdm

        return _tqdm
    except Exception:
        pass

    try:
        from tqdm import tqdm as _tqdm

        return _tqdm
    except Exception:
        return None


def _as_json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _to_bool_flag(value: Any) -> int:
    return 1 if bool(value) else 0


def _normalize_outcome(result: Any) -> str | None:
    if result is None:
        return None
    text = str(result).strip().lower()
    if text in {"yes", "no"}:
        return text
    return None


def _normalize_market_row(
    market: dict[str, Any],
    *,
    source_kind: str,
    event_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ticker = str(market.get("ticker") or "").strip()
    event_ticker = str(market.get("event_ticker") or (event_row or {}).get("event_ticker") or "").strip()
    series_ticker = str((event_row or {}).get("series_ticker") or market.get("series_ticker") or "").strip() or None
    title = market.get("title") or (event_row or {}).get("title")
    event_title = (event_row or {}).get("title")
    event_sub_title = (event_row or {}).get("sub_title")
    subtitle = market.get("subtitle") or market.get("sub_title")
    rules_primary = market.get("rules_primary")
    rules_secondary = market.get("rules_secondary")
    result = _normalize_outcome(market.get("result"))
    expiration_time = market.get("expiration_time")
    close_time = market.get("close_time")

    final_yes_probability: float | None
    if result == "yes":
        final_yes_probability = 1.0
    elif result == "no":
        final_yes_probability = 0.0
    else:
        final_yes_probability = None

    description = rules_primary or event_sub_title or subtitle
    status = market.get("status")
    status_norm = str(status).strip().lower() if status is not None else ""
    is_active = status_norm in {"active", "open"}
    is_resolved = status_norm in {"finalized", "settled"} or result in {"yes", "no"}

    return {
        "market_id": f"kalshi:{ticker}" if ticker else None,
        "source": "kalshi",
        "venue_market_id": ticker or None,
        "event_id": f"kalshi:event:{event_ticker}" if event_ticker else None,
        "venue_event_id": event_ticker or None,
        "series_ticker": series_ticker,
        "ticker": ticker or None,
        "event_ticker": event_ticker or None,
        "title": title,
        "question": title,
        "subtitle": subtitle,
        "yes_sub_title": market.get("yes_sub_title"),
        "no_sub_title": market.get("no_sub_title"),
        "market_type": market.get("market_type"),
        "status": status,
        "event_title": event_title,
        "event_sub_title": event_sub_title,
        "kalshi_category": (event_row or {}).get("category"),
        "mutually_exclusive": _to_bool_flag((event_row or {}).get("mutually_exclusive")),
        "strike_period": (event_row or {}).get("strike_period"),
        "rules_primary": rules_primary,
        "rules_secondary": rules_secondary,
        "can_close_early": _to_bool_flag(market.get("can_close_early")),
        "early_close_condition": market.get("early_close_condition"),
        "is_provisional": _to_bool_flag(market.get("is_provisional")),
        "result": result,
        "settlement_value_dollars": _to_float(market.get("settlement_value_dollars")),
        "created_at": market.get("created_time"),
        "updated_at": market.get("updated_time") or (event_row or {}).get("last_updated_ts"),
        "open_time": market.get("open_time"),
        "close_time": close_time,
        "expected_expiration_time": market.get("expected_expiration_time"),
        "expiration_time": expiration_time,
        "latest_expiration_time": market.get("latest_expiration_time"),
        "settlement_ts": market.get("settlement_ts"),
        "last_price_dollars": _to_float(market.get("last_price_dollars")),
        "previous_price_dollars": _to_float(market.get("previous_price_dollars")),
        "yes_bid_dollars": _to_float(market.get("yes_bid_dollars")),
        "yes_ask_dollars": _to_float(market.get("yes_ask_dollars")),
        "no_bid_dollars": _to_float(market.get("no_bid_dollars")),
        "no_ask_dollars": _to_float(market.get("no_ask_dollars")),
        "yes_bid_size_fp": _to_float(market.get("yes_bid_size_fp")),
        "yes_ask_size_fp": _to_float(market.get("yes_ask_size_fp")),
        "volume_num": _to_float(market.get("volume_fp")),
        "volume_24h_num": _to_float(market.get("volume_24h_fp")),
        "open_interest_num": _to_float(market.get("open_interest_fp")),
        "liquidity_dollars": _to_float(market.get("liquidity_dollars")),
        "notional_value_dollars": _to_float(market.get("notional_value_dollars")),
        "response_price_units": market.get("response_price_units"),
        "price_level_structure": market.get("price_level_structure"),
        "tick_size": market.get("tick_size"),
        "strike_type": market.get("strike_type"),
        "floor_strike": _to_float(market.get("floor_strike")),
        "cap_strike": _to_float(market.get("cap_strike")),
        "functional_strike": market.get("functional_strike"),
        "custom_strike_json": _as_json_text(market.get("custom_strike")),
        "mve_collection_ticker": market.get("mve_collection_ticker"),
        "mve_selected_legs_json": _as_json_text(market.get("mve_selected_legs")),
        "description": description,
        "end_date": expiration_time or close_time,
        "final_outcome": result,
        "final_yes_probability": final_yes_probability,
        "is_binary": _to_bool_flag(str(market.get("market_type") or "").strip().lower() == "binary"),
        "is_resolved": _to_bool_flag(is_resolved),
        "is_active": _to_bool_flag(is_active),
        "is_closed": _to_bool_flag(not is_active),
        "data_source_kind": source_kind,
    }


def _format_seen_ts(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return value.strftime("%Y-%m-%d")


class MarketsCollector:
    def __init__(self, client: KalshiClient) -> None:
        self._client = client

    @staticmethod
    def _estimate_total_pages(max_pages: int | None, *, estimate_total: bool) -> int | None:
        if not estimate_total:
            return None
        if max_pages is not None:
            return int(max_pages)
        return None

    @staticmethod
    def _chunked(iterable: Iterable[dict[str, Any]], *, chunk_size: int) -> Iterable[list[dict[str, Any]]]:
        chunk: list[dict[str, Any]] = []
        for item in iterable:
            chunk.append(item)
            if len(chunk) >= int(chunk_size):
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    @staticmethod
    def _date_window(rows: list[dict[str, Any]]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not rows:
            return None, None
        created = pd.to_datetime(pd.Series([row.get("created_at") for row in rows]), utc=True, errors="coerce").dropna()
        if created.empty:
            return None, None
        return created.min(), created.max()

    @staticmethod
    def _prepare_index_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        out["created_at_sort"] = pd.to_datetime(out.get("created_at"), utc=True, errors="coerce")
        out = out.sort_values(
            ["market_id", "created_at_sort"],
            ascending=[True, False],
            kind="stable",
        )
        out = out.drop_duplicates(subset=["market_id"], keep="first").reset_index(drop=True)
        out = out.drop(columns=["created_at_sort"], errors="ignore")
        out["indexed_at_utc"] = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%dT%H:%M:%SZ")
        for column in _RAW_MARKETS_COLUMNS:
            if column not in out.columns:
                out[column] = None
        return out[_RAW_MARKETS_COLUMNS].copy()

    def iter_market_index_batches(
        self,
        *,
        endpoint: IndexEndpoint = "markets",
        limit: int = 200,
        max_pages: int | None = None,
        show_progress: bool = True,
        estimate_total: bool = True,
        **params: Any,
    ) -> Iterable[list[dict[str, Any]]]:
        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")
        total = self._estimate_total_pages(max_pages, estimate_total=estimate_total)
        pbar = (
            tqdm(
                total=total,
                disable=False,
                unit="page",
                desc=(
                    "Downloading Kalshi raw markets"
                    if endpoint == "markets"
                    else "Downloading Kalshi historical raw markets"
                ),
                leave=True,
            )
            if tqdm is not None
            else None
        )
        pages = 0
        market_rows_total = 0
        seen_min: pd.Timestamp | None = None
        seen_max: pd.Timestamp | None = None
        try:
            if endpoint == "historical":
                source_iter = self._client.iter_historical_markets(limit=limit, max_pages=max_pages, **params)
                source_kind = "markets_historical_index"
            else:
                source_iter = self._client.iter_markets(limit=limit, max_pages=max_pages, **params)
                source_kind = "markets_live_index"
            for batch in self._chunked(source_iter, chunk_size=max(1, int(limit))):
                rows = [
                    _normalize_market_row(market, source_kind=source_kind, event_row=None)
                    for market in batch
                    if isinstance(market, dict)
                ]
                pages += 1
                market_rows_total += len(rows)
                batch_min, batch_max = self._date_window(rows)
                if batch_min is not None:
                    seen_min = batch_min if seen_min is None else min(seen_min, batch_min)
                if batch_max is not None:
                    seen_max = batch_max if seen_max is None else max(seen_max, batch_max)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "pages": pages,
                            "rows": market_rows_total,
                            "seen_min_created": _format_seen_ts(seen_min),
                            "seen_max_created": _format_seen_ts(seen_max),
                        }
                    )
                yield rows
        finally:
            if pbar is not None:
                pbar.close()
