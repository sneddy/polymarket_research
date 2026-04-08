from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import csv
import io
import logging
from typing import Any
from zipfile import ZipFile

from clients.binance_client import BinanceClient
from config import BinanceDataConfig, HttpConfig
from utils import ensure_datetime_utc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


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


class BinanceArchiveClient:
    """Downloader for historical kline ZIP archives hosted at data.binance.vision."""

    def __init__(
        self,
        *,
        binance_data: BinanceDataConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None and session is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._binance_data = binance_data or BinanceDataConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/zip,application/octet-stream,*/*")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    def download_klines(
        self,
        symbol: str,
        *,
        interval: str,
        start_date: datetime | str,
        end_date: datetime | str,
        tail_days: int = 45,
        show_progress: bool = True,
    ) -> list[list[str]]:
        start_dt = ensure_datetime_utc(start_date)
        end_dt = ensure_datetime_utc(end_date)
        if end_dt <= start_dt:
            raise ValueError("end_date must be > start_date")

        rows: list[list[str]] = []
        seen_open_time_ms: set[int] = set()

        monthly_items = self._iter_month_starts(start_dt.date(), end_dt.date())
        tail_start = max(start_dt.date(), end_dt.date() - timedelta(days=max(0, int(tail_days))))
        daily_items = self._iter_days(tail_start, end_dt.date())
        total_files = len(monthly_items) + len(daily_items)

        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")
        pbar = (
            tqdm(
                total=total_files,
                disable=False,
                unit="file",
                desc=f"Binance archive {symbol.upper()} {interval}",
                leave=True,
            )
            if tqdm is not None
            else None
        )

        try:
            for month_start in monthly_items:
                url = self.monthly_klines_url(symbol, interval=interval, year=month_start.year, month=month_start.month)
                added = 0
                for row in self._download_zip_rows_if_exists(url):
                    open_time_ms = self._extract_open_time_ms(row)
                    if open_time_ms is None or open_time_ms in seen_open_time_ms:
                        continue
                    seen_open_time_ms.add(open_time_ms)
                    rows.append(row)
                    added += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"phase": "monthly", "period": month_start.isoformat()[:7], "rows": len(rows), "added": added})

            for day in daily_items:
                url = self.daily_klines_url(symbol, interval=interval, day=day)
                added = 0
                for row in self._download_zip_rows_if_exists(url):
                    open_time_ms = self._extract_open_time_ms(row)
                    if open_time_ms is None or open_time_ms in seen_open_time_ms:
                        continue
                    seen_open_time_ms.add(open_time_ms)
                    rows.append(row)
                    added += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"phase": "daily", "period": day.isoformat(), "rows": len(rows), "added": added})
        finally:
            if pbar is not None:
                pbar.close()

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        filtered = []
        for row in rows:
            open_time_ms = self._extract_open_time_ms(row)
            if open_time_ms is None:
                continue
            if start_ms <= open_time_ms < end_ms:
                filtered.append(row)

        filtered.sort(key=lambda row: int(row[0]))
        return filtered

    def monthly_klines_url(self, symbol: str, *, interval: str, year: int, month: int) -> str:
        name = f"{symbol.upper()}-{interval}-{year:04d}-{month:02d}.zip"
        return (
            self._binance_data.base_url.rstrip("/")
            + f"/spot/monthly/klines/{symbol.upper()}/{interval}/{name}"
        )

    def daily_klines_url(self, symbol: str, *, interval: str, day: date) -> str:
        name = f"{symbol.upper()}-{interval}-{day.isoformat()}.zip"
        return (
            self._binance_data.base_url.rstrip("/")
            + f"/spot/daily/klines/{symbol.upper()}/{interval}/{name}"
        )

    def _download_zip_rows_if_exists(self, url: str) -> list[list[str]]:
        resp = self._session.get(url, timeout=self._http.timeout_seconds)
        if resp.status_code == 404:
            logger.debug("Binance archive not found | url=%s", url)
            return []
        resp.raise_for_status()
        with ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                return []
            with zf.open(names[0], "r") as fh:
                text = fh.read().decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if row]
        return rows

    @staticmethod
    def _extract_open_time_ms(row: list[str]) -> int | None:
        if not row:
            return None
        try:
            return BinanceClient._normalize_epoch_ms(row[0])
        except Exception:
            return None

    @staticmethod
    def _iter_month_starts(start_day: date, end_day: date) -> list[date]:
        cur = date(start_day.year, start_day.month, 1)
        end_month = date(end_day.year, end_day.month, 1)
        out: list[date] = []
        while cur <= end_month:
            out.append(cur)
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)
        return out

    @staticmethod
    def _iter_days(start_day: date, end_day: date) -> list[date]:
        out: list[date] = []
        cur = start_day
        while cur <= end_day:
            out.append(cur)
            cur += timedelta(days=1)
        return out
