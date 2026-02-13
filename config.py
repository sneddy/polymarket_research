from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class HttpConfig:
    timeout_seconds: float = 30.0
    max_retries: int = 5
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 30.0
    user_agent: str = "polymarket_research/0.1"


@dataclass(frozen=True)
class GammaConfig:
    base_url: str = "https://gamma-api.polymarket.com"


@dataclass(frozen=True)
class DataApiConfig:
    base_url: str = "https://data-api.polymarket.com"
    # As of 2025-08-26 Polymarket capped /trades pagination (see official changelog).
    max_limit: int = 500
    max_offset: int = 1000


@dataclass(frozen=True)
class ClobConfig:
    rest_base_url: str = "https://clob.polymarket.com"
    ws_market_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass(frozen=True)
class OrderbookSubgraphConfig:
    """
    Public GraphQL endpoints for Polymarket's orderbook subgraph.

    Used for full-history trades downloads (avoids Data API pagination caps).
    """

    endpoints: tuple[str, ...] = (
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn",
        # Some older links reference a versioned endpoint; keep as fallback.
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn",
    )


@dataclass(frozen=True)
class NewsConfig:
    # Default: GDELT 2.1 DOC API (no key required).
    gdelt_doc_base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc"


def load_http_config_from_env(prefix: str = "POLY_") -> HttpConfig:
    def _get_float(key: str, default: float) -> float:
        val = os.getenv(prefix + key)
        return default if val is None else float(val)

    def _get_int(key: str, default: int) -> int:
        val = os.getenv(prefix + key)
        return default if val is None else int(val)

    user_agent = os.getenv(prefix + "USER_AGENT", HttpConfig.user_agent)

    return HttpConfig(
        timeout_seconds=_get_float("HTTP_TIMEOUT_SECONDS", HttpConfig.timeout_seconds),
        max_retries=_get_int("HTTP_MAX_RETRIES", HttpConfig.max_retries),
        backoff_base_seconds=_get_float("HTTP_BACKOFF_BASE_SECONDS", HttpConfig.backoff_base_seconds),
        backoff_max_seconds=_get_float("HTTP_BACKOFF_MAX_SECONDS", HttpConfig.backoff_max_seconds),
        user_agent=user_agent,
    )
