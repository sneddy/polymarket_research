from __future__ import annotations

from datetime import UTC, datetime
import csv
import io
import logging
from typing import Any

from config import FredConfig, HttpConfig
from utils import ensure_datetime_utc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class FredClient:
    """Minimal CSV-based client for FRED series downloads."""

    def __init__(
        self,
        *,
        fred: FredConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None and session is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._fred = fred or FredConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "text/csv")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    def download_series_csv(
        self,
        series_id: str,
        *,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> list[dict[str, Any]]:
        start_dt = ensure_datetime_utc(start_date)
        end_dt = ensure_datetime_utc(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        params = {
            "id": str(series_id),
            "cosd": start_dt.date().isoformat(),
            "coed": end_dt.date().isoformat(),
        }
        resp = self._session.get(self._fred.graph_csv_url, params=params, timeout=self._http.timeout_seconds)
        resp.raise_for_status()
        if "observation_date" not in resp.text:
            raise ValueError(f"Unexpected FRED CSV response for series_id={series_id!r}")

        reader = csv.DictReader(io.StringIO(resp.text))
        out: list[dict[str, Any]] = []
        value_column = None if not reader.fieldnames else next((c for c in reader.fieldnames if c != "observation_date"), None)
        if value_column is None:
            return out

        for row in reader:
            raw_date = row.get("observation_date")
            raw_value = row.get(value_column)
            if raw_date is None or raw_value in (None, "", "."):
                continue
            dt = datetime.fromisoformat(raw_date).replace(tzinfo=UTC)
            try:
                value = float(raw_value)
            except Exception:
                continue
            out.append(
                {
                    "timestamp_utc": dt,
                    "value": value,
                    "close": value,
                    "open": value,
                    "high": value,
                    "low": value,
                    "volume": None,
                }
            )
        return out
