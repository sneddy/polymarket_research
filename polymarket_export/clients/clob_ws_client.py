from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

try:
    import websockets
except Exception:  # pragma: no cover - optional dependency
    websockets = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class ClobWsClient:
    """Small async WebSocket wrapper for Polymarket CLOB market feeds."""

    def __init__(
        self,
        ws_url: str,
        *,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        open_timeout: float | None = 10.0,
        close_timeout: float | None = 10.0,
    ) -> None:
        self._ws_url = ws_url
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout
        self._close_timeout = close_timeout
        self._conn: Any | None = None

    @property
    def ws_url(self) -> str:
        return self._ws_url

    @property
    def connected(self) -> bool:
        return self._conn is not None

    async def connect(self) -> None:
        if websockets is None:
            raise ImportError("Missing dependency: websockets. Install with `pip install websockets`.")

        if self._conn is not None:
            return

        self._conn = await websockets.connect(
            self._ws_url,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            open_timeout=self._open_timeout,
            close_timeout=self._close_timeout,
        )

    async def close(self) -> None:
        if self._conn is None:
            return
        try:
            await self._conn.close()
        finally:
            self._conn = None

    async def send_json(self, message: dict[str, Any]) -> None:
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected; call connect() first.")
        await self._conn.send(json.dumps(message, separators=(",", ":")))

    async def recv_json(self, *, timeout_seconds: float | None = None) -> Any:
        if self._conn is None:
            raise RuntimeError("WebSocket is not connected; call connect() first.")

        if timeout_seconds is None:
            raw = await self._conn.recv()
        else:
            raw = await asyncio.wait_for(self._conn.recv(), timeout=timeout_seconds)

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)

    async def subscribe_market(self, token_ids: list[str], **extra: Any) -> None:
        """
        Subscribe to Polymarket market channel for given token ids.

        Docs example:
          {"type":"market","assets_ids":["<tokenId>"]}
        """

        message = {"type": "market", "assets_ids": token_ids, **extra}
        await self.send_json(message)

