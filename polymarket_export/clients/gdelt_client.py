from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import logging
import random
import re
import time
from typing import Any, Iterable

from config import HttpConfig, NewsConfig
from utils import ensure_datetime_utc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_DOC_PAGE_SIZE = 250
_TIMESPAN_RE = re.compile(r"^(?P<num>\d+)(?P<unit>[smhdw])$", re.IGNORECASE)


def _dt_to_gdelt(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y%m%d%H%M%S")


def _parse_retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        n = float(s)
    except Exception:
        return None
    if n <= 0:
        return None
    return n


def _parse_timespan_to_timedelta(timespan: str) -> timedelta | None:
    s = str(timespan).strip().lower()
    m = _TIMESPAN_RE.fullmatch(s)
    if m is None:
        return None

    qty = int(m.group("num"))
    unit = m.group("unit")
    if qty <= 0:
        return None

    if unit == "s":
        return timedelta(seconds=qty)
    if unit == "m":
        return timedelta(minutes=qty)
    if unit == "h":
        return timedelta(hours=qty)
    if unit == "d":
        return timedelta(days=qty)
    if unit == "w":
        return timedelta(weeks=qty)
    return None


class GDELTClient:
    """
    Minimal, retry-aware wrapper for GDELT 2.1 DOC API.

    Supports:
      - full-text query
      - start/end bounded windows
      - timespan-based windows
      - backward pagination by timestamp cursor (datedesc)
      - safe raw JSON responses
    """

    def __init__(
        self,
        *,
        news: NewsConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._news = news or NewsConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    def search_doc_raw(
        self,
        query: str,
        *,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        timespan: str | None = None,
        max_records: int = 250,
        page_size: int = 250,
        sort: str = "datedesc",
    ) -> dict[str, Any]:
        """
        Fetch raw DOC API payloads with pagination and aggregate into one raw bundle.

        Returns:
          {
            "query": ...,
            "articles": [...],
            "pages": [...],
            "article_count": int,
            "page_count": int,
          }
        """

        pages: list[dict[str, Any]] = []
        articles: list[dict[str, Any]] = []

        for payload in self.iter_doc_pages(
            query,
            start_date=start_date,
            end_date=end_date,
            timespan=timespan,
            max_records=max_records,
            page_size=page_size,
            sort=sort,
        ):
            pages.append(payload)
            batch = self.extract_articles(payload)
            if batch:
                articles.extend(batch)
            if len(articles) >= int(max_records):
                break

        if len(articles) > int(max_records):
            articles = articles[: int(max_records)]

        return {
            "query": query,
            "articles": articles,
            "pages": pages,
            "article_count": int(len(articles)),
            "page_count": int(len(pages)),
        }

    def iter_doc_pages(
        self,
        query: str,
        *,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        timespan: str | None = None,
        max_records: int = 250,
        page_size: int = 250,
        sort: str = "datedesc",
    ) -> Iterable[dict[str, Any]]:
        """
        Yield raw DOC API page payloads.

        Pagination strategy:
          - Request with `sort=datedesc` in a [start, cursor_end] time window.
          - Move cursor_end to oldest_seen_timestamp - 1s.
          - Stop on empty page, no cursor progress, or max_records reached.

        If `timespan` is provided in an unknown format (cannot parse to timedelta),
        one page is fetched using raw `timespan` and pagination stops.
        """

        q = str(query).strip()
        if not q:
            raise ValueError("query must be non-empty")

        total_target = int(max_records)
        if total_target <= 0:
            return

        size = int(page_size)
        if size <= 0:
            raise ValueError("page_size must be positive")
        size = min(size, _MAX_DOC_PAGE_SIZE)

        start_dt, end_dt, raw_timespan = self._resolve_time_bounds(
            start_date=start_date,
            end_date=end_date,
            timespan=timespan,
        )

        fetched = 0
        cursor_end = end_dt

        while fetched < total_target:
            req_limit = min(size, total_target - fetched)
            payload = self._request_doc_page(
                query=q,
                start_dt=start_dt,
                end_dt=cursor_end,
                timespan=raw_timespan,
                max_records=req_limit,
                sort=sort,
            )
            yield payload

            articles = self.extract_articles(payload)
            if not articles:
                return

            fetched += len(articles)
            if fetched >= total_target:
                return

            # If we had to pass raw timespan directly, we cannot safely cursor-page.
            if raw_timespan is not None:
                return

            oldest = self.oldest_article_datetime(articles)
            if oldest is None:
                return

            next_end = oldest - timedelta(seconds=1)
            if cursor_end is not None and next_end >= cursor_end:
                return
            if start_dt is not None and next_end < start_dt:
                return

            cursor_end = next_end

    @staticmethod
    def extract_articles(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            articles = payload.get("articles")
            if isinstance(articles, list):
                return [a for a in articles if isinstance(a, dict)]
        return []

    @staticmethod
    def parse_article_datetime(article: dict[str, Any]) -> datetime | None:
        for key in ("seendate", "seenDate", "published", "publishedAt", "date"):
            if key not in article:
                continue
            dt = GDELTClient.parse_timestamp_value(article.get(key))
            if dt is not None:
                return dt
        return None

    @staticmethod
    def parse_timestamp_value(value: Any) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            return ensure_datetime_utc(value)

        if isinstance(value, (int, float)):
            ts = int(value)
            # ms epoch support
            if ts > 10**12:
                ts = int(ts / 1000)
            try:
                return datetime.fromtimestamp(ts, tz=UTC)
            except Exception:
                return None

        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            if s.isdigit() and len(s) == 14:
                try:
                    dt = datetime.strptime(s, "%Y%m%d%H%M%S")
                    return dt.replace(tzinfo=UTC)
                except Exception:
                    return None
            if s.isdigit():
                try:
                    ts = int(s)
                    if ts > 10**12:
                        ts = int(ts / 1000)
                    return datetime.fromtimestamp(ts, tz=UTC)
                except Exception:
                    return None
            try:
                return ensure_datetime_utc(s)
            except Exception:
                return None

        return None

    @staticmethod
    def oldest_article_datetime(articles: list[dict[str, Any]]) -> datetime | None:
        oldest: datetime | None = None
        for article in articles:
            dt = GDELTClient.parse_article_datetime(article)
            if dt is None:
                continue
            if oldest is None or dt < oldest:
                oldest = dt
        return oldest

    def _resolve_time_bounds(
        self,
        *,
        start_date: datetime | str | None,
        end_date: datetime | str | None,
        timespan: str | None,
    ) -> tuple[datetime | None, datetime | None, str | None]:
        start_dt = ensure_datetime_utc(start_date) if start_date is not None else None
        end_dt = ensure_datetime_utc(end_date) if end_date is not None else None
        raw_timespan: str | None = None

        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        if timespan is not None and (start_dt is None or end_dt is None):
            parsed_delta = _parse_timespan_to_timedelta(timespan)
            if parsed_delta is not None:
                if end_dt is None:
                    end_dt = datetime.now(tz=UTC)
                if start_dt is None:
                    start_dt = end_dt - parsed_delta
            elif start_dt is None and end_dt is None:
                raw_timespan = str(timespan)

        # For pagination, pin open-ended queries to "now".
        if raw_timespan is None and end_dt is None:
            end_dt = datetime.now(tz=UTC)

        if start_dt is not None and end_dt is not None and end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        return start_dt, end_dt, raw_timespan

    def _request_doc_page(
        self,
        *,
        query: str,
        start_dt: datetime | None,
        end_dt: datetime | None,
        timespan: str | None,
        max_records: int,
        sort: str,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": sort,
            "maxrecords": int(max_records),
        }

        if start_dt is not None:
            params["startdatetime"] = _dt_to_gdelt(start_dt)
        if end_dt is not None:
            params["enddatetime"] = _dt_to_gdelt(end_dt)
        if timespan is not None and start_dt is None and end_dt is None:
            params["timespan"] = timespan

        payload = self._get_json(self._news.gdelt_doc_base_url, params=params)
        if isinstance(payload, dict):
            return payload
        return {"articles": []}

    def _get_json(self, url: str, *, params: dict[str, Any]) -> Any:
        attempts = max(1, int(self._http.max_retries) + 1)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            resp = self._request("GET", url, params=params)
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                snippet = (resp.text or "")[:500]
                last_exc = ValueError(f"Non-JSON response from GDELT: {snippet}")
                if attempt >= attempts - 1:
                    raise last_exc from e
                logger.warning(
                    "GDELT returned non-JSON payload (attempt %s/%s); retrying.",
                    attempt + 1,
                    attempts,
                )
                self._sleep_before_retry(attempt, retry_after_seconds=None)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Failed to parse JSON response from GDELT.")

    def _request(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> "requests.Response":
        attempts = max(1, int(self._http.max_retries) + 1)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=float(self._http.timeout_seconds),
                )
            except requests.RequestException as e:  # type: ignore[attr-defined]
                last_exc = e
                if attempt >= attempts - 1:
                    raise
                self._sleep_before_retry(attempt, retry_after_seconds=None)
                continue

            if resp.status_code in _RETRY_STATUS_CODES and attempt < attempts - 1:
                retry_after = _parse_retry_after_seconds(resp.headers.get("Retry-After"))
                logger.warning(
                    "GDELT request retryable status %s (attempt %s/%s).",
                    resp.status_code,
                    attempt + 1,
                    attempts,
                )
                self._sleep_before_retry(attempt, retry_after_seconds=retry_after)
                continue

            resp.raise_for_status()
            return resp

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Request failed without response.")

    def _sleep_before_retry(self, attempt: int, *, retry_after_seconds: float | None) -> None:
        exp = float(self._http.backoff_base_seconds) * (2 ** int(attempt))
        delay = min(float(self._http.backoff_max_seconds), exp)
        jitter = random.uniform(0.0, min(1.0, delay * 0.25))
        delay = delay + jitter
        if retry_after_seconds is not None:
            delay = max(delay, float(retry_after_seconds))
        time.sleep(max(0.0, delay))
