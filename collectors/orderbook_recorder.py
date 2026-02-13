from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
from typing import Any, Literal

from clients.clob_ws_client import ClobWsClient
from clients.gamma_client import GammaClient
from config import ClobConfig, HttpConfig
from storage.parquet_store import ParquetStore
from utils import pick_condition_id_from_markets

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


def _is_condition_id(value: str) -> bool:
    return isinstance(value, str) and value.startswith("0x") and len(value) >= 10


class OrderBookRecorder:
    """
    Notebook-friendly CLOB order book recorder.

    - No background threads.
    - Use synchronous wrappers in scripts, or await the async methods in notebooks.
    """

    def __init__(
        self,
        *,
        clob: ClobConfig | None = None,
        http: HttpConfig | None = None,
        gamma_client: GammaClient | None = None,
        store: ParquetStore | None = None,
        session: "requests.Session | None" = None,
    ) -> None:
        if requests is None:
            raise ImportError("Missing dependency: requests. Install with `pip install requests`.")

        self._clob = clob or ClobConfig()
        self._http = http or HttpConfig()
        self._gamma = gamma_client
        self._store = store or ParquetStore()

        self._ws = ClobWsClient(self._clob.ws_market_url)
        self._session = session or requests.Session()
        self._session.headers.setdefault("Accept", "application/json")
        self._session.headers.setdefault("User-Agent", self._http.user_agent)

        self._subscribed_token_ids: list[str] = []

    # -----------------------
    # Sync wrappers (scripts)
    # -----------------------
    def connect(self) -> None:
        self._run(self.aconnect())

    def subscribe(self, market_id: str) -> list[str]:
        return self._run(self.asubscribe(market_id))

    def subscribe_url(self, url: str, *, market_index: int | None = None, market_slug: str | None = None) -> list[str]:
        return self._run(self.asubscribe_url(url, market_index=market_index, market_slug=market_slug))

    def get_snapshot(self, token_id: str | None = None) -> Any:
        return self._run(self.aget_snapshot(token_id))

    def record(self, duration_seconds: int, *, max_messages: int | None = None, frame_type: FrameType | None = None) -> Any:
        return self._run(self.arecord(duration_seconds, max_messages=max_messages, frame_type=frame_type))

    def close(self) -> None:
        self._run(self.aclose())

    def save_to_parquet(self, df: Any, path: str, *, partition_cols: list[str] | None = None) -> str:
        out = self._store.save(df, path, partition_cols=partition_cols)
        return str(out)

    # -----------------------
    # Async (notebooks)
    # -----------------------
    async def aconnect(self) -> None:
        await self._ws.connect()

    async def aclose(self) -> None:
        await self._ws.close()

    async def asubscribe(self, market_id: str) -> list[str]:
        token_ids = await self._resolve_token_ids(market_id)
        await self._ws.subscribe_market(token_ids)
        self._subscribed_token_ids = token_ids
        return token_ids

    async def asubscribe_url(self, url: str, *, market_index: int | None = None, market_slug: str | None = None) -> list[str]:
        if self._gamma is None:
            raise ValueError("Provide gamma_client to resolve Polymarket URLs.")
        markets = self._gamma.resolve_markets_from_polymarket_url(url)
        condition_id = pick_condition_id_from_markets(markets, market_index=market_index, market_slug=market_slug)
        return await self.asubscribe(condition_id)

    async def aget_snapshot(self, token_id: str | None = None) -> Any:
        if token_id is not None:
            return self._fetch_book(token_id)

        if not self._subscribed_token_ids:
            raise ValueError("No token_id provided and nothing subscribed yet.")

        return {tid: self._fetch_book(tid) for tid in self._subscribed_token_ids}

    async def arecord(
        self,
        duration_seconds: int,
        *,
        max_messages: int | None = None,
        frame_type: FrameType | None = None,
    ) -> Any:
        if not self._ws.connected:
            await self._ws.connect()

        rows: list[dict[str, Any]] = []
        deadline = asyncio.get_running_loop().time() + float(duration_seconds)

        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break

            try:
                msg = await self._ws.recv_json(timeout_seconds=min(remaining, 5.0))
            except TimeoutError:
                continue
            except asyncio.TimeoutError:
                continue

            received_at = datetime.now(tz=UTC)
            if isinstance(msg, dict):
                event_type = msg.get("event_type") or msg.get("eventType")
                asset_id = msg.get("asset_id") or msg.get("assetId")
            else:
                event_type = None
                asset_id = None

            rows.append(
                {
                    "received_at": received_at,
                    "event_type": event_type,
                    "asset_id": asset_id,
                    "message": json.dumps(msg, separators=(",", ":"), ensure_ascii=False),
                }
            )

            if max_messages is not None and len(rows) >= max_messages:
                break

        return self._to_frame(rows, frame_type=frame_type)

    # -----------------------
    # Internals
    # -----------------------
    async def _resolve_token_ids(self, market_id: str) -> list[str]:
        if not _is_condition_id(market_id):
            return [market_id]

        if self._gamma is None:
            raise ValueError("market_id looks like a condition id (0x...). Provide gamma_client to resolve token ids.")

        markets = self._gamma.get_markets(active=True, limit=10, condition_ids=[market_id])
        if isinstance(markets, dict):
            markets = markets.get("markets") or markets.get("data") or markets.get("results") or markets.get("items") or []
        if not isinstance(markets, list) or not markets:
            raise ValueError(f"Could not resolve condition id via Gamma: {market_id}")

        m0 = markets[0] if isinstance(markets[0], dict) else {}
        token_ids = (
            m0.get("clobTokenIds")
            or m0.get("clob_token_ids")
            or m0.get("clobTokenIDs")
            or m0.get("clob_token_IDs")
        )
        if not token_ids:
            raise ValueError("Gamma market payload did not include clobTokenIds/clob_token_ids.")
        if isinstance(token_ids, str):
            token_ids = [token_ids]
        if not isinstance(token_ids, list):
            raise ValueError(f"Unexpected token id type: {type(token_ids)!r}")

        return [str(t) for t in token_ids]

    def _fetch_book(self, token_id: str) -> Any:
        url = self._clob.rest_base_url.rstrip("/") + "/book"
        resp = self._session.get(url, params={"token_id": token_id}, timeout=self._http.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _to_frame(rows: list[dict[str, Any]], *, frame_type: FrameType | None) -> Any:
        frame = frame_type or _default_frame_type()
        if frame == "pandas":
            import pandas as pd

            return pd.DataFrame(rows)
        if frame == "polars":
            import polars as pl

            return pl.DataFrame(rows)
        raise ValueError(f"Unknown frame_type: {frame!r}")

    @staticmethod
    def _run(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("This method cannot run inside an existing event loop. Use the async (a*) methods instead.")
