from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime
import re
from typing import Any
from urllib.parse import urlparse, unquote


_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_POLY_PATH = re.compile(r"^/(event|market)/([^/?#]+)")
_TRAILING_NUM = re.compile(r"^(?P<base>.+?)-(?P<num>\d+)$")


def camel_to_snake(name: str) -> str:
    name = _CAMEL_1.sub(r"\1_\2", name)
    name = _CAMEL_2.sub(r"\1_\2", name)
    return name.replace("-", "_").lower()


def to_snake_case_keys(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {camel_to_snake(str(k)): to_snake_case_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_snake_case_keys(v) for v in obj]
    return obj


def ensure_datetime_utc(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        # Accept ISO8601-like strings; best-effort parsing.
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError(f"Unsupported date type: {type(value)!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def chunked(seq: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_polymarket_market_or_event_url(url: str) -> tuple[str, str]:
    """
    Extract ("event"|"market", slug) from a Polymarket URL.

    Examples:
      https://polymarket.com/event/fed-decision-in-october?tid=...
      https://polymarket.com/market/will-bitcoin-hit-100k-by-2025
    """

    parsed = urlparse(url)
    if not parsed.netloc:
        raise ValueError(f"Not a URL: {url!r}")

    m = _POLY_PATH.match(parsed.path or "")
    if not m:
        raise ValueError(f"Unsupported Polymarket URL path: {parsed.path!r}")

    kind = m.group(1)
    slug = unquote(m.group(2))
    if not slug:
        raise ValueError("Could not extract slug from URL")
    return kind, slug


def slug_variants(slug: str) -> list[str]:
    """
    Return candidate slugs to try against Gamma.

    Polymarket sometimes shares links like `<slug>-<id>`. Gamma slug endpoints
    typically use the base slug without the numeric suffix.
    """

    out = [slug]
    m = _TRAILING_NUM.match(slug)
    if m:
        base = m.group("base")
        if base and base not in out:
            out.append(base)
    return out


def pick_condition_id_from_markets(
    markets: list[Mapping[str, Any]],
    *,
    market_index: int | None = None,
    market_slug: str | None = None,
) -> str:
    """
    Pick a single condition id from a list of Gamma market dicts.

    Intended for Polymarket /event/ URLs that resolve to multiple markets.
    """

    if not markets:
        raise ValueError("No markets to choose from.")

    def _get_slug(m: Mapping[str, Any]) -> str | None:
        s = m.get("slug")
        return None if s is None else str(s)

    def _get_cid(m: Mapping[str, Any]) -> str | None:
        cid = m.get("conditionId") or m.get("condition_id")
        return None if cid is None else str(cid)

    if market_slug is not None:
        for m in markets:
            if _get_slug(m) == market_slug:
                cid = _get_cid(m)
                if cid:
                    return cid
                raise ValueError(f"Selected market_slug={market_slug!r} has no conditionId.")

        options = [(i, _get_slug(m), _get_cid(m)) for i, m in enumerate(markets)]
        raise ValueError(f"market_slug not found: {market_slug!r}. Options: {options}")

    if market_index is not None:
        idx = int(market_index)
        if idx < 0 or idx >= len(markets):
            options = [(i, _get_slug(m), _get_cid(m)) for i, m in enumerate(markets)]
            raise ValueError(f"market_index out of range (0..{len(markets)-1}). Options: {options}")
        cid = _get_cid(markets[idx])
        if cid:
            return cid
        raise ValueError(f"Selected market_index={idx} has no conditionId.")

    if len(markets) == 1:
        cid = _get_cid(markets[0])
        if cid:
            return cid
        raise ValueError("Resolved a single market but it has no conditionId.")

    options = [(i, _get_slug(m), _get_cid(m)) for i, m in enumerate(markets)]
    raise ValueError(f"Multiple markets resolved; choose one via market_index or market_slug. Options: {options}")
