"""Reusable market-level exclusion filters for Polymarket registry selection."""

from __future__ import annotations

from typing import Any


SHORT_HORIZON_UPDOWN_PATTERNS = (
    "-updown-5m-",
    "-updown-15m-",
    "-updown-4h-",
)


def is_short_horizon_updown_series(
    *,
    slug: Any,
    event_slug: Any,
) -> bool:
    """Detect recurring short-horizon up/down markets by slug pattern."""
    slug_text = str(slug or "").lower()
    event_slug_text = str(event_slug or "").lower()
    return any(
        pattern in slug_text or pattern in event_slug_text
        for pattern in SHORT_HORIZON_UPDOWN_PATTERNS
    )
