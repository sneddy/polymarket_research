from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import random
import time
from typing import Any, Iterable

from clients.gamma_client import GammaClient
from config import ClobConfig, HttpConfig
from storage.sqlite_orderbook_store import SqliteOrderBookStore
from utils import parse_polymarket_market_or_event_url

try:
    import requests
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_LEVELS = 10


@dataclass(frozen=True)
class MarketOutcome:
    source_url: str
    source_kind: str
    source_slug: str
    market_id: str | None
    market_slug: str
    market_question: str | None
    group_item_title: str | None
    condition_id: str
    outcome_index: int
    outcome_name: str
    token_id: str
    active: bool
    closed: bool
    archived: bool
    enable_order_book: bool
    updated_at_utc: str


@dataclass(frozen=True)
class BookLevel:
    side: str
    level_index: int
    price: float
    size: float


@dataclass(frozen=True)
class OutcomeSnapshot:
    token_id: str
    condition_id: str
    market_slug: str
    outcome_name: str
    captured_at_utc: str
    book_timestamp_ms: int | None
    book_hash: str | None
    best_bid: float | None
    best_ask: float | None
    last_trade_price: float | None
    min_order_size: float | None
    tick_size: float | None
    bid_levels: list[BookLevel]
    ask_levels: list[BookLevel]


@dataclass(frozen=True)
class PollError:
    token_id: str
    condition_id: str
    market_slug: str
    outcome_name: str
    error_message: str


@dataclass(frozen=True)
class PollResult:
    source_url: str
    captured_at_utc: str
    interval_seconds: float
    levels_requested: int
    outcomes_expected: int
    snapshots: list[OutcomeSnapshot]
    errors: list[PollError]


class OrderBookSnapshotCollector:
    """Reliable polling collector for limited CLOB order book snapshots."""

    def __init__(
        self,
        *,
        clob: ClobConfig | None = None,
        http: HttpConfig | None = None,
        gamma_client: GammaClient | None = None,
        store: SqliteOrderBookStore | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None and (gamma_client is None or session is None):
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._clob = clob or ClobConfig()
        self._http = http or HttpConfig()
        self._gamma = gamma_client or GammaClient(http=self._http)
        self._store = store or SqliteOrderBookStore()

        if session is None:
            assert requests is not None
            session = requests.Session()
        self._session = session
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

    def resolve_market_outcomes(
        self,
        url: str,
        *,
        include_closed: bool = False,
    ) -> list[MarketOutcome]:
        source_kind, source_slug = parse_polymarket_market_or_event_url(url)
        markets = self._gamma.resolve_markets_from_polymarket_url(url)

        resolved: list[MarketOutcome] = []
        seen_token_ids: set[str] = set()
        updated_at_utc = datetime.now(tz=UTC).isoformat()

        for market in markets:
            if not isinstance(market, dict):
                continue

            closed = _coerce_bool(market.get("closed"))
            archived = _coerce_bool(market.get("archived"))
            enable_order_book = _coerce_bool(market.get("enableOrderBook"), default=True)
            active = _coerce_bool(market.get("active"), default=False)

            if not include_closed and (closed or archived):
                continue
            if not enable_order_book:
                continue

            condition_id = _coerce_str(market.get("conditionId") or market.get("condition_id"))
            market_slug = _coerce_str(market.get("slug"))
            if condition_id is None or market_slug is None:
                continue

            outcomes = _parse_list(market.get("outcomes"))
            token_ids = _parse_list(market.get("clobTokenIds") or market.get("clob_token_ids"))
            if not outcomes or not token_ids:
                continue
            if len(outcomes) != len(token_ids):
                raise ValueError(
                    f"Outcome/token mismatch for market {market_slug!r}: "
                    f"{len(outcomes)} outcomes vs {len(token_ids)} token ids."
                )

            for idx, (outcome_name, token_id) in enumerate(zip(outcomes, token_ids, strict=True)):
                token_id_str = str(token_id)
                if token_id_str in seen_token_ids:
                    raise ValueError(f"Duplicate token id while resolving URL {url!r}: {token_id_str}")
                seen_token_ids.add(token_id_str)

                resolved.append(
                    MarketOutcome(
                        source_url=url,
                        source_kind=source_kind,
                        source_slug=source_slug,
                        market_id=_coerce_str(market.get("id")),
                        market_slug=market_slug,
                        market_question=_coerce_str(market.get("question")),
                        group_item_title=_coerce_str(market.get("groupItemTitle") or market.get("group_item_title")),
                        condition_id=condition_id,
                        outcome_index=idx,
                        outcome_name=str(outcome_name),
                        token_id=token_id_str,
                        active=active,
                        closed=closed,
                        archived=archived,
                        enable_order_book=enable_order_book,
                        updated_at_utc=updated_at_utc,
                    )
                )

        if not resolved:
            raise ValueError(f"No open order-book outcomes resolved from URL: {url}")

        logger.info(
            "Resolved %s outcomes from %s (%s)",
            len(resolved),
            source_kind,
            url,
        )
        for outcome in resolved:
            logger.info(
                "Resolved outcome: condition_id=%s token_id=%s market_slug=%s outcome=%s",
                outcome.condition_id,
                outcome.token_id,
                outcome.market_slug,
                outcome.outcome_name,
            )

        return resolved

    def poll_once(
        self,
        *,
        source_url: str,
        outcomes: list[MarketOutcome],
        interval_seconds: float,
        levels: int,
    ) -> PollResult:
        levels = _validated_levels(levels)
        captured_at_utc = datetime.now(tz=UTC).isoformat()

        logger.info(
            "Starting poll cycle: captured_at_utc=%s outcomes=%s levels=%s interval_seconds=%s",
            captured_at_utc,
            len(outcomes),
            levels,
            interval_seconds,
        )

        snapshots: list[OutcomeSnapshot] = []
        errors: list[PollError] = []

        for outcome in outcomes:
            try:
                book = self._fetch_book(outcome.token_id)
                snapshots.append(
                    self._normalize_snapshot(
                        outcome=outcome,
                        book=book,
                        captured_at_utc=captured_at_utc,
                        levels=levels,
                    )
                )
                logger.info(
                    "Snapshot fetched: captured_at_utc=%s condition_id=%s token_id=%s market_slug=%s outcome=%s book_timestamp_ms=%s",
                    captured_at_utc,
                    outcome.condition_id,
                    outcome.token_id,
                    outcome.market_slug,
                    outcome.outcome_name,
                    snapshots[-1].book_timestamp_ms,
                )
            except Exception as exc:
                logger.warning(
                    "Snapshot failed: captured_at_utc=%s condition_id=%s token_id=%s market_slug=%s outcome=%s error=%s",
                    captured_at_utc,
                    outcome.condition_id,
                    outcome.token_id,
                    outcome.market_slug,
                    outcome.outcome_name,
                    exc,
                )
                errors.append(
                    PollError(
                        token_id=outcome.token_id,
                        condition_id=outcome.condition_id,
                        market_slug=outcome.market_slug,
                        outcome_name=outcome.outcome_name,
                        error_message=str(exc),
                    )
                )

        result = PollResult(
            source_url=source_url,
            captured_at_utc=captured_at_utc,
            interval_seconds=float(interval_seconds),
            levels_requested=levels,
            outcomes_expected=len(outcomes),
            snapshots=snapshots,
            errors=errors,
        )
        logger.info(
            "Finished poll cycle: captured_at_utc=%s succeeded=%s failed=%s",
            captured_at_utc,
            len(snapshots),
            len(errors),
        )
        return result

    def run(
        self,
        *,
        url: str,
        db_path: str,
        interval_seconds: float = 10.0,
        levels: int = 10,
        max_polls: int | None = None,
        include_closed: bool = False,
    ) -> list[PollResult]:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")

        outcomes = self.resolve_market_outcomes(url, include_closed=include_closed)
        self._store.upsert_market_outcomes(db_path, outcomes)
        logger.info(
            "Market outcomes stored: db_path=%s outcomes=%s",
            db_path,
            len(outcomes),
        )

        results: list[PollResult] = []
        polls_done = 0
        next_started_at = time.monotonic()

        while max_polls is None or polls_done < max_polls:
            result = self.poll_once(
                source_url=url,
                outcomes=outcomes,
                interval_seconds=interval_seconds,
                levels=levels,
            )
            poll_cycle_id = self._store.append_poll_result(db_path, result)
            logger.info(
                "Poll cycle stored: poll_cycle_id=%s captured_at_utc=%s db_path=%s succeeded=%s failed=%s",
                poll_cycle_id,
                result.captured_at_utc,
                db_path,
                len(result.snapshots),
                len(result.errors),
            )
            results.append(result)
            polls_done += 1

            if max_polls is not None and polls_done >= max_polls:
                break

            next_started_at += float(interval_seconds)
            sleep_seconds = next_started_at - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                next_started_at = time.monotonic()

        return results

    def _fetch_book(self, token_id: str) -> Any:
        url = self._clob.rest_base_url.rstrip("/") + "/book"
        resp = self._request("GET", url, params={"token_id": token_id})
        payload = resp.json()

        asset_id = _coerce_str(payload.get("asset_id") or payload.get("assetId"))
        if asset_id is not None and asset_id != token_id:
            raise ValueError(f"Book response token mismatch: requested={token_id}, got={asset_id}")

        return payload

    def _request(self, method: str, url: str, *, params: dict[str, Any] | None = None) -> "requests.Response":
        assert requests is not None

        attempts = max(1, int(self._http.max_retries) + 1)
        last_exc: Exception | None = None

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
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                self._sleep_backoff(attempt, None)

        raise RuntimeError("Unreachable") from last_exc

    def _sleep_backoff(self, attempt: int, resp: "requests.Response | None") -> None:
        retry_after = None
        if resp is not None:
            raw_retry_after = resp.headers.get("Retry-After")
            if raw_retry_after is not None:
                try:
                    retry_after = float(raw_retry_after)
                except Exception:
                    retry_after = None

        base = float(self._http.backoff_base_seconds) * (2 ** max(0, attempt - 1))
        jitter = random.random() * base * 0.25
        delay = min(base + jitter, float(self._http.backoff_max_seconds))
        if retry_after is not None:
            delay = min(max(delay, retry_after), float(self._http.backoff_max_seconds))
        time.sleep(delay)

    @staticmethod
    def _normalize_snapshot(
        *,
        outcome: MarketOutcome,
        book: dict[str, Any],
        captured_at_utc: str,
        levels: int,
    ) -> OutcomeSnapshot:
        bid_levels = _normalize_levels(book.get("bids"), side="bid", levels=levels)
        ask_levels = _normalize_levels(book.get("asks"), side="ask", levels=levels)

        return OutcomeSnapshot(
            token_id=outcome.token_id,
            condition_id=outcome.condition_id,
            market_slug=outcome.market_slug,
            outcome_name=outcome.outcome_name,
            captured_at_utc=captured_at_utc,
            book_timestamp_ms=_coerce_int(book.get("timestamp")),
            book_hash=_coerce_str(book.get("hash")),
            best_bid=bid_levels[0].price if bid_levels else None,
            best_ask=ask_levels[0].price if ask_levels else None,
            last_trade_price=_coerce_float(book.get("last_trade_price") or book.get("lastTradePrice")),
            min_order_size=_coerce_float(book.get("min_order_size") or book.get("minOrderSize")),
            tick_size=_coerce_float(book.get("tick_size") or book.get("tickSize")),
            bid_levels=bid_levels,
            ask_levels=ask_levels,
        )


def _normalize_levels(levels_raw: Any, *, side: str, levels: int) -> list[BookLevel]:
    levels = _validated_levels(levels)
    if side not in {"bid", "ask"}:
        raise ValueError(f"Unsupported side: {side!r}")

    normalized: list[tuple[float, float]] = []
    for row in _iter_levels(levels_raw):
        price = _coerce_float(row.get("price"))
        size = _coerce_float(row.get("size"))
        if price is None or size is None:
            continue
        normalized.append((price, size))

    normalized.sort(key=lambda item: item[0], reverse=(side == "bid"))
    trimmed = normalized[:levels]

    return [
        BookLevel(side=side, level_index=index, price=price, size=size)
        for index, (price, size) in enumerate(trimmed)
    ]


def _iter_levels(levels_raw: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(levels_raw, list):
        return []
    return [row for row in levels_raw if isinstance(row, dict)]


def _parse_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"Expected list or JSON list string, got {type(value)!r}")


def _validated_levels(levels: int) -> int:
    val = int(levels)
    if val < 1 or val > _MAX_LEVELS:
        raise ValueError(f"levels must be in [1, {_MAX_LEVELS}]")
    return val


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
