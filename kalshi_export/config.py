from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _load_simple_dotenv(path: Path) -> None:
    """Load a simple KEY=VALUE `.env` file if present."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_simple_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class HttpConfig:
    timeout_seconds: float = 30.0
    max_retries: int = 6
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    series_pause_seconds: float = 0.2
    user_agent: str = "polymarket_research/0.1"


@dataclass(frozen=True)
class KalshiApiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    api_key: str | None = None


def load_http_config_from_env(prefix: str = "KALSHI_") -> HttpConfig:
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
        series_pause_seconds=_get_float("HTTP_SERIES_PAUSE_SECONDS", HttpConfig.series_pause_seconds),
        user_agent=user_agent,
    )


def load_kalshi_api_config_from_env(prefix: str = "KALSHI_") -> KalshiApiConfig:
    return KalshiApiConfig(
        base_url=os.getenv(prefix + "BASE_URL", KalshiApiConfig.base_url),
        api_key=os.getenv(prefix + "API_KEY"),
    )
