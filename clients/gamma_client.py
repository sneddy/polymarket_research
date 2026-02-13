from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
import json
import logging
import random
import time
from typing import Any, TypeVar
from urllib.parse import urljoin
from urllib.parse import quote

from config import DataApiConfig, GammaConfig, HttpConfig
from utils import parse_polymarket_market_or_event_url, slug_variants

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

T = TypeVar("T")


_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _coerce_bool(value: bool) -> str:
    return "true" if value else "false"


class GammaClient:
    """
    Lightweight wrapper around Polymarket REST APIs used in research:

    - Gamma Markets API (metadata): https://gamma-api.polymarket.com
    - Data API (historical trades): https://data-api.polymarket.com

    Methods return raw JSON for maximum debuggability.
    """

    def __init__(
        self,
        *,
        gamma: GammaConfig | None = None,
        data_api: DataApiConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._gamma = gamma or GammaConfig()
        self._data_api = data_api or DataApiConfig()
        self._http = http or HttpConfig()

        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    @property
    def gamma_base_url(self) -> str:
        return self._gamma.base_url.rstrip("/") + "/"

    @property
    def data_api_base_url(self) -> str:
        return self._data_api.base_url.rstrip("/") + "/"

    def get_markets(self, active: bool = True, limit: int = 100, offset: int = 0, **params: Any) -> Any:
        url = urljoin(self.gamma_base_url, "markets")
        q = {"active": _coerce_bool(active), "limit": limit, "offset": offset, **params}
        return self._get_json(url, params=q)

    def get_market_by_slug(self, slug: str) -> Any:
        url = urljoin(self.gamma_base_url, f"markets/slug/{quote(slug)}")
        return self._get_json(url, params={})

    def get_event_by_slug(self, slug: str) -> Any:
        url = urljoin(self.gamma_base_url, f"events/slug/{quote(slug)}")
        return self._get_json(url, params={})

    def get_trades(self, market_id: str, limit: int = 1000, offset: int = 0, **params: Any) -> Any:
        """
        Historical trades via Polymarket Data API.

        Notes:
        - Data API currently caps `limit` and `offset` (see config.DataApiConfig).
        - `market_id` here is a condition id (0x...).
        """

        url = urljoin(self.data_api_base_url, "trades")

        effective_limit = min(int(limit), int(self._data_api.max_limit))
        if effective_limit != limit:
            logger.debug("Clamped Data API trades limit from %s to %s", limit, effective_limit)

        if offset > self._data_api.max_offset:
            raise ValueError(
                f"Data API trades offset {offset} exceeds configured max_offset={self._data_api.max_offset}. "
                "Polymarket caps this; consider filtering by time and collecting incrementally."
            )

        q = {"market": market_id, "limit": effective_limit, "offset": int(offset), **params}
        return self._get_json(url, params=q)

    def resolve_condition_ids_from_polymarket_url(self, url: str) -> list[str]:
        """
        Resolve condition id(s) from a Polymarket event/market URL.

        - /market/<slug> -> single conditionId
        - /event/<slug>  -> potentially multiple conditionIds
        """

        markets = self.resolve_markets_from_polymarket_url(url)
        ids: list[str] = []
        for m in markets:
            if not isinstance(m, dict):
                continue
            cid = m.get("conditionId") or m.get("condition_id")
            if cid:
                ids.append(str(cid))
        if not ids:
            raise ValueError("Resolved markets did not include conditionId")
        return ids

    def resolve_markets_from_polymarket_url(self, url: str) -> list[dict[str, Any]]:
        """Resolve market object(s) from a Polymarket event/market URL via Gamma slug endpoints."""

        kind, slug = parse_polymarket_market_or_event_url(url)
        candidates = slug_variants(slug)

        if kind == "market":
            for s in candidates:
                try:
                    market = self.get_market_by_slug(s)
                except requests.HTTPError as e:  # type: ignore[attr-defined]
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise
                if isinstance(market, dict):
                    return [market]
            raise ValueError(f"Could not resolve market slug via Gamma: {slug!r}")

        if kind == "event":
            last_event: Any | None = None
            for s in candidates:
                try:
                    event = self.get_event_by_slug(s)
                    last_event = event
                except requests.HTTPError as e:  # type: ignore[attr-defined]
                    if e.response is not None and e.response.status_code == 404:
                        continue
                    raise
                markets = event.get("markets") if isinstance(event, dict) else None
                if isinstance(markets, list):
                    return [m for m in markets if isinstance(m, dict)]
            if last_event is not None:
                raise ValueError("Gamma event response did not include a markets list")
            raise ValueError(f"Could not resolve event slug via Gamma: {slug!r}")

        raise ValueError(f"Unsupported Polymarket URL type: {kind!r}")

    def iter_markets(
        self,
        *,
        active: bool = True,
        limit: int = 100,
        start_offset: int = 0,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        def _fetch(*, limit: int, offset: int) -> Any:
            return self.get_markets(active=active, limit=limit, offset=offset, **params)

        yield from self._iter_offset_pages(_fetch, limit=limit, start_offset=start_offset, max_pages=max_pages)

    def estimate_markets_count(
        self,
        *,
        active: bool = True,
        max_probe_offset: int = 4_000_000,
        **params: Any,
    ) -> int | None:
        """
        Estimate total number of markets matching filters.

        Gamma's `/markets` endpoint does not expose an official total-count field/header.
        This method probes `limit=1` pages and uses exponential + binary search over `offset`
        to find the first empty page, which corresponds to the total count.

        Returns None if probing fails.
        """

        def _has_item(offset: int) -> bool:
            payload = self.get_markets(active=active, limit=1, offset=int(offset), **params)
            items = self._extract_list(payload)
            return bool(items)

        try:
            if not _has_item(0):
                return 0
        except Exception:
            return None

        lo = 0
        hi = 1
        try:
            while hi <= int(max_probe_offset) and _has_item(hi):
                lo = hi
                hi *= 2
        except Exception:
            return None

        if hi > int(max_probe_offset):
            return None

        # Binary search first empty offset in (lo, hi].
        left = lo + 1
        right = hi
        try:
            while left < right:
                mid = (left + right) // 2
                if _has_item(mid):
                    left = mid + 1
                else:
                    right = mid
        except Exception:
            return None

        first_empty = left
        return int(first_empty)

    def iter_trades(
        self,
        market_id: str,
        *,
        limit: int = 500,
        start_offset: int = 0,
        max_pages: int | None = None,
        **params: Any,
    ) -> Iterable[dict[str, Any]]:
        def _fetch(*, limit: int, offset: int) -> Any:
            return self.get_trades(market_id, limit=limit, offset=offset, **params)

        yield from self._iter_offset_pages(_fetch, limit=limit, start_offset=start_offset, max_pages=max_pages)

    def _iter_offset_pages(
        self,
        fetch: Callable[..., Any],
        *,
        limit: int,
        start_offset: int,
        max_pages: int | None,
    ) -> Iterable[dict[str, Any]]:
        offset = int(start_offset)
        pages = 0
        while True:
            payload = fetch(limit=limit, offset=offset)
            items = self._extract_list(payload)
            if not items:
                return

            for item in items:
                if isinstance(item, dict):
                    yield item
                else:
                    yield {"value": item}

            pages += 1
            if max_pages is not None and pages >= max_pages:
                return
            offset += len(items)

    @staticmethod
    def _extract_list(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("data", "markets", "results", "items"):
                val = payload.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _get_json(self, url: str, *, params: dict[str, Any]) -> Any:
        resp = self._request("GET", url, params=params)
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            snippet = (resp.text or "")[:500]
            raise ValueError(f"Non-JSON response from {url}: {snippet}") from e

    def _request(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> "requests.Response":
        assert requests is not None

        last_exc: Exception | None = None
        attempts = max(1, int(self._http.max_retries) + 1)

        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self._http.timeout_seconds,
                )
                if resp.status_code in _RETRY_STATUS_CODES and attempt < attempts:
                    self._sleep_backoff(attempt, resp)
                    continue

                resp.raise_for_status()
                return resp
            except Exception as e:  # requests exceptions + HTTPError
                last_exc = e
                if attempt >= attempts:
                    raise
                self._sleep_backoff(attempt, None)

        raise RuntimeError("Unreachable") from last_exc

    def _sleep_backoff(self, attempt: int, resp: "requests.Response | None") -> None:
        retry_after = None
        if resp is not None:
            ra = resp.headers.get("Retry-After")
            if ra:
                try:
                    retry_after = float(ra)
                except Exception:
                    retry_after = None

        base = float(self._http.backoff_base_seconds) * (2 ** max(0, attempt - 1))
        jitter = random.random() * base * 0.25
        delay = base + jitter
        if retry_after is not None:
            delay = max(delay, retry_after)
        delay = min(delay, float(self._http.backoff_max_seconds))
        time.sleep(delay)
