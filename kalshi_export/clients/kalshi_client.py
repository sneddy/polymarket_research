from __future__ import annotations

from collections.abc import Iterable
import logging
import random
import time
from typing import Any, TypeVar
from urllib.parse import urljoin

from config import HttpConfig, KalshiApiConfig, load_http_config_from_env, load_kalshi_api_config_from_env

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class KalshiClient:
    """Lightweight wrapper around Kalshi REST APIs used by `kalshi_export`."""

    def __init__(
        self,
        *,
        api: KalshiApiConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._api = api or load_kalshi_api_config_from_env()
        self._http = http or load_http_config_from_env()

        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    @property
    def base_url(self) -> str:
        return self._api.base_url.rstrip("/") + "/"

    def get_events(
        self,
        *,
        limit: int = 200,
        cursor: str | None = None,
        with_nested_markets: bool = True,
        **params: Any,
    ) -> Any:
        q = {"limit": int(limit), "with_nested_markets": str(with_nested_markets).lower(), **params}
        if cursor:
            q["cursor"] = cursor
        return self._get_json("events", params=q)

    def get_event(self, event_ticker: str, *, with_nested_markets: bool = False, **params: Any) -> Any:
        q = {"with_nested_markets": str(with_nested_markets).lower(), **params}
        return self._get_json(f"events/{event_ticker}", params=q)

    def get_markets(self, *, limit: int = 200, cursor: str | None = None, **params: Any) -> Any:
        q = {"limit": int(limit), **params}
        if cursor:
            q["cursor"] = cursor
        return self._get_json("markets", params=q)

    def get_historical_markets(self, *, limit: int = 200, cursor: str | None = None, **params: Any) -> Any:
        q = {"limit": int(limit), **params}
        if cursor:
            q["cursor"] = cursor
        return self._get_json("historical/markets", params=q)

    def iter_events(
        self,
        *,
        limit: int = 200,
        max_pages: int | None = None,
        with_nested_markets: bool = True,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        yield from self._iter_cursor_pages(
            "events",
            list_key="events",
            limit=limit,
            max_pages=max_pages,
            with_nested_markets=str(with_nested_markets).lower(),
            **params,
        )

    def iter_historical_markets(
        self,
        *,
        limit: int = 200,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        yield from self._iter_cursor_pages(
            "historical/markets",
            list_key="markets",
            limit=limit,
            max_pages=max_pages,
            **params,
        )

    def iter_markets(
        self,
        *,
        limit: int = 200,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        yield from self._iter_cursor_pages(
            "markets",
            list_key="markets",
            limit=limit,
            max_pages=max_pages,
            **params,
        )

    def _iter_cursor_pages(
        self,
        path: str,
        *,
        list_key: str,
        limit: int,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        pages = 0
        cursor: str | None = None
        while True:
            q = {"limit": int(limit), **params}
            if cursor:
                q["cursor"] = cursor
            payload = self._get_json(path, params=q)
            items = payload.get(list_key) if isinstance(payload, dict) else None
            if not isinstance(items, list) or not items:
                break
            for item in items:
                if isinstance(item, dict):
                    yield item
            pages += 1
            if max_pages is not None and pages >= int(max_pages):
                break
            next_cursor = payload.get("cursor") if isinstance(payload, dict) else None
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)

    def _get_json(self, path: str, *, params: dict[str, Any]) -> Any:
        url = urljoin(self.base_url, path)
        return self._request_json(url, params=params)

    def _request_json(self, url: str, *, params: dict[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._http.max_retries + 1):
            try:
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._http.timeout_seconds,
                )
                if response.status_code in _RETRY_STATUS_CODES:
                    response.raise_for_status()
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                should_retry = status_code in _RETRY_STATUS_CODES or status_code is None
                if attempt >= self._http.max_retries or not should_retry:
                    raise
                sleep_seconds = min(
                    self._http.backoff_max_seconds,
                    self._http.backoff_base_seconds * (2**attempt) * (1.0 + random.random()),
                )
                logger.warning(
                    "Kalshi request failed; retrying | attempt=%s url=%s status_code=%s sleep_seconds=%.2f error=%s",
                    attempt + 1,
                    url,
                    status_code,
                    sleep_seconds,
                    exc,
                )
                time.sleep(sleep_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Kalshi request failed without a captured error for url={url}")
