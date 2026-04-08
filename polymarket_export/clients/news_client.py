from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any, Literal

from config import HttpConfig, NewsConfig
from utils import ensure_datetime_utc, to_snake_case_keys

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

FrameType = Literal["pandas", "polars"]


def _default_frame_type() -> FrameType:
    try:
        import pandas as _  # noqa: F401

        return "pandas"
    except Exception:
        return "polars"


def _dt_to_gdelt(dt: datetime) -> str:
    # GDELT expects YYYYMMDDHHMMSS (UTC).
    return dt.strftime("%Y%m%d%H%M%S")


class NewsClient:
    """
    News search client.

    Default backend: GDELT 2.1 DOC API (no key required).
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

    def search_news(
        self,
        query: str,
        start_date: datetime | str,
        end_date: datetime | str,
        *,
        max_records: int = 250,
        frame_type: FrameType | None = None,
        raw: bool = False,
    ) -> Any:
        """
        Search for news articles matching a query in a date range.

        Returns a pandas/polars DataFrame (default) unless `raw=True`.
        """

        start_dt = ensure_datetime_utc(start_date)
        end_dt = ensure_datetime_utc(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": int(max_records),
            "startdatetime": _dt_to_gdelt(start_dt),
            "enddatetime": _dt_to_gdelt(end_dt),
            "sort": "datedesc",
        }

        resp = self._session.get(self._news.gdelt_doc_base_url, params=params, timeout=self._http.timeout_seconds)
        resp.raise_for_status()
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            snippet = (resp.text or "")[:500]
            raise ValueError(f"Non-JSON response from GDELT: {snippet}") from e

        if raw:
            return payload

        articles = payload.get("articles", []) if isinstance(payload, dict) else []
        rows = [to_snake_case_keys(a) for a in articles if isinstance(a, dict)]
        return self._to_frame(rows, frame_type=frame_type)

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        if frame == "pandas":
            import pandas as pd

            df = pd.DataFrame(rows)
            if "seendate" in df.columns:
                col = df["seendate"].astype("string")
                parsed_ymd = pd.to_datetime(col, format="%Y%m%d%H%M%S", errors="coerce", utc=True)
                parsed_iso = pd.to_datetime(col.str.replace("Z", "+00:00", regex=False), errors="coerce", utc=True)
                df["seendate"] = parsed_ymd.fillna(parsed_iso)
            return df

        if frame == "polars":
            import polars as pl

            df = pl.DataFrame(rows)
            if "seendate" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("seendate").cast(pl.Utf8).str.len_chars() == 14)
                    .then(pl.col("seendate").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d%H%M%S", strict=False))
                    .otherwise(pl.col("seendate").cast(pl.Utf8).str.replace("Z$", "+00:00").str.strptime(pl.Datetime, strict=False))
                    .dt.replace_time_zone("UTC")
                    .alias("seendate")
                )
            return df

        raise ValueError(f"Unknown frame_type: {frame!r}")
