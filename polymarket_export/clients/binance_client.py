from __future__ import annotations

from datetime import UTC, datetime
import logging
from typing import Any, Iterable

from config import BinanceConfig, HttpConfig
from utils import ensure_datetime_utc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_KLINE_COLUMNS = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_asset_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


class BinanceClient:
    """Small REST wrapper for Binance kline downloads."""

    def __init__(
        self,
        *,
        binance: BinanceConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._binance = binance or BinanceConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    def iter_klines(
        self,
        symbol: str,
        *,
        interval: str,
        start_date: datetime | str,
        end_date: datetime | str,
        limit: int | None = None,
    ) -> Iterable[list[Any]]:
        start_dt = ensure_datetime_utc(start_date)
        end_dt = ensure_datetime_utc(end_date)
        if end_dt <= start_dt:
            raise ValueError("end_date must be > start_date")

        page_limit = min(int(limit or self._binance.max_klines_limit), int(self._binance.max_klines_limit))
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        while start_ms < end_ms:
            params = {
                "symbol": str(symbol).upper(),
                "interval": str(interval),
                "startTime": int(start_ms),
                "endTime": int(end_ms),
                "limit": page_limit,
            }
            url = self._binance.rest_base_url.rstrip("/") + "/api/v3/klines"
            resp = self._session.get(url, params=params, timeout=self._http.timeout_seconds)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or not payload:
                return

            for row in payload:
                if isinstance(row, list):
                    yield row

            last = payload[-1]
            if not isinstance(last, list) or len(last) < 7:
                return

            next_start_ms = int(last[6]) + 1
            if next_start_ms <= start_ms:
                return
            start_ms = next_start_ms

    @staticmethod
    def normalize_kline_row(row: list[Any]) -> dict[str, Any]:
        payload = {key: row[idx] if idx < len(row) else None for idx, key in enumerate(_KLINE_COLUMNS)}
        open_time_ms = BinanceClient._normalize_epoch_ms(payload["open_time_ms"])
        close_time_ms = BinanceClient._normalize_epoch_ms(payload["close_time_ms"])
        return {
            "timestamp_utc": datetime.fromtimestamp(open_time_ms / 1000.0, tz=UTC),
            "close_timestamp_utc": datetime.fromtimestamp(close_time_ms / 1000.0, tz=UTC),
            "open": float(payload["open"]),
            "high": float(payload["high"]),
            "low": float(payload["low"]),
            "close": float(payload["close"]),
            "volume": float(payload["volume"]),
            "quote_asset_volume": float(payload["quote_asset_volume"]),
            "trade_count": int(payload["trade_count"]),
            "taker_buy_base_volume": float(payload["taker_buy_base_volume"]),
            "taker_buy_quote_volume": float(payload["taker_buy_quote_volume"]),
        }

    @staticmethod
    def _normalize_epoch_ms(value: Any) -> int:
        ts = int(value)
        # Some Binance archive dumps encode these timestamps in microseconds.
        if ts >= 10**15:
            return ts // 1000
        return ts
