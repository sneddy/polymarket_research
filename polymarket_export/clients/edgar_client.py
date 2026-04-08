from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import logging
import re
from typing import Any
from xml.etree import ElementTree as ET

from config import HttpConfig, SecEdgarConfig
from utils import ensure_datetime_utc

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_INDEX_ROW_RE = re.compile(
    r"^(?P<form_type>\S+)\s+(?P<company_name>.*?)\s+(?P<cik>\d+)\s+(?P<date_filed>\d{8})\s+(?P<file_name>\S+)\s*$"
)


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


class EdgarClient:
    """Downloader for daily SEC EDGAR form-index files."""

    def __init__(
        self,
        *,
        edgar: SecEdgarConfig | None = None,
        http: HttpConfig | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None and session is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._edgar = edgar or SecEdgarConfig()
        self._http = http or HttpConfig()
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "text/plain, text/*;q=0.9, */*;q=0.1")
        # requests.Session ships with its own default User-Agent, but SEC expects
        # callers to declare a descriptive identifier with contact info.
        self._session.headers["User-Agent"] = self._edgar.user_agent or self._http.user_agent
        self._daily_index_cache: dict[date, list[dict[str, Any]]] = {}
        self._warn_if_user_agent_looks_generic()

    def download_daily_form_counts(
        self,
        form_types: str | list[str] | tuple[str, ...],
        *,
        start_date: datetime | str,
        end_date: datetime | str,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        start_dt = ensure_datetime_utc(start_date)
        end_dt = ensure_datetime_utc(end_date)
        if end_dt < start_dt:
            raise ValueError("end_date must be >= start_date")

        selected = self._normalize_form_filter(form_types)
        out: list[dict[str, Any]] = []
        days = self._iter_days(start_dt.date(), end_dt.date())
        tqdm = _resolve_tqdm(show_progress=show_progress)
        if show_progress and tqdm is None:
            logger.warning("show_progress=True but tqdm is unavailable in the active environment.")

        form_label = "all forms" if selected is None else ",".join(sorted(selected))
        pbar = (
            tqdm(
                total=len(days),
                disable=False,
                unit="day",
                desc=f"SEC EDGAR {form_label}",
                leave=False,
            )
            if tqdm is not None
            else None
        )

        try:
            for day in days:
                entries = self.download_form_index(day)
                count = sum(1 for row in entries if self._form_matches(row.get("form_type"), selected))
                dt = datetime(day.year, day.month, day.day, tzinfo=UTC)
                out.append(
                    {
                        "timestamp_utc": dt,
                        "value": float(count),
                        "close": float(count),
                        "open": float(count),
                        "high": float(count),
                        "low": float(count),
                        "volume": None,
                    }
                )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"day": day.isoformat(), "count": count})
        finally:
            if pbar is not None:
                pbar.close()
        return out

    def download_form_index(self, day: date | datetime | str) -> list[dict[str, Any]]:
        target = ensure_datetime_utc(day).date()
        cached = self._daily_index_cache.get(target)
        if cached is not None:
            return list(cached)
        url = self.daily_form_index_url(target)
        resp = self._session.get(url, timeout=self._http.timeout_seconds)
        if resp.status_code == 404:
            logger.debug("SEC EDGAR form index not found | day=%s url=%s", target.isoformat(), url)
            return []
        if resp.status_code == 403 and self._is_missing_index_response(resp.text or ""):
            logger.debug("SEC EDGAR form index missing (403 XML) | day=%s url=%s", target.isoformat(), url)
            self._daily_index_cache[target] = []
            return []
        self._raise_if_sec_blocked(resp.text or "")
        resp.raise_for_status()
        parsed = self.parse_form_index_text(resp.text, fallback_day=target)
        self._daily_index_cache[target] = parsed
        return list(parsed)

    def daily_form_index_url(self, day: date) -> str:
        quarter = ((int(day.month) - 1) // 3) + 1
        return (
            self._edgar.daily_index_base_url.rstrip("/")
            + f"/{day.year}/QTR{quarter}/form.{day.strftime('%Y%m%d')}.idx"
        )

    @staticmethod
    def parse_form_index_text(text: str, *, fallback_day: date | None = None) -> list[dict[str, Any]]:
        lines = text.splitlines()
        header_end = -1
        for i, line in enumerate(lines):
            if line.strip().startswith("---"):
                header_end = i
                break

        body = lines[header_end + 1 :] if header_end >= 0 else lines
        out: list[dict[str, Any]] = []
        for raw_line in body:
            line = raw_line.rstrip()
            if not line.strip():
                continue
            m = _INDEX_ROW_RE.match(line)
            if m is None:
                continue
            filed_day = datetime.strptime(m.group("date_filed"), "%Y%m%d").date()
            out.append(
                {
                    "form_type": m.group("form_type").strip(),
                    "company_name": m.group("company_name").strip(),
                    "cik": m.group("cik").strip(),
                    "date_filed": filed_day if filed_day is not None else fallback_day,
                    "file_name": m.group("file_name").strip(),
                }
            )
        return out

    def _warn_if_user_agent_looks_generic(self) -> None:
        ua = str(self._session.headers.get("User-Agent") or "").strip()
        if not ua:
            logger.warning("SEC EDGAR requests should set a descriptive User-Agent with contact info.")
            return
        if "@" not in ua and "(" not in ua:
            logger.warning(
                "SEC EDGAR requests work best with a declared User-Agent including contact info. "
                "Configure SecEdgarConfig.user_agent in config.py before using sec_edgar."
            )

    @staticmethod
    def _normalize_form_filter(form_types: str | list[str] | tuple[str, ...]) -> set[str] | None:
        if isinstance(form_types, str):
            raw_items = [part.strip() for part in form_types.split(",")]
        else:
            raw_items = [str(part).strip() for part in form_types]

        normalized = {item.upper() for item in raw_items if item}
        if not normalized or "*" in normalized:
            return None
        return normalized

    @staticmethod
    def _form_matches(form_type: Any, selected: set[str] | None) -> bool:
        if selected is None:
            return True
        if form_type is None:
            return False
        return str(form_type).strip().upper() in selected

    @staticmethod
    def _raise_if_sec_blocked(text: str) -> None:
        if "Undeclared Automated Tool" in text:
            raise RuntimeError(
                "SEC blocked the request because the User-Agent is not sufficiently declared. "
                "Update SecEdgarConfig.user_agent in config.py to a descriptive value with contact info before using sec_edgar."
            )

    @staticmethod
    def _is_missing_index_response(text: str) -> bool:
        raw = (text or "").strip()
        if not raw or not raw.startswith("<"):
            return False
        try:
            root = ET.fromstring(raw)
        except Exception:
            return False
        code = (root.findtext("Code") or "").strip()
        message = (root.findtext("Message") or "").strip().lower()
        return code in {"AccessDenied", "NoSuchKey"} or "does not exist" in message or "access denied" in message

    @staticmethod
    def _iter_days(start_day: date, end_day: date) -> list[date]:
        out: list[date] = []
        cur = start_day
        while cur <= end_day:
            out.append(cur)
            cur += timedelta(days=1)
        return out
