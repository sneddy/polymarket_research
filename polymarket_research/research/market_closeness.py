"""Template surface for future market-closeness research modules.

The closeness analysis is still notebook-first and intentionally remains in
`frozen_notebooks/2_ticker_closeness.ipynb` for iteration. This module exists
only to reserve the package surface for a later, better-specified extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MarketClosenessConfig:
    """Placeholder config for a future extracted closeness pipeline."""

    top_k: int = 5


@dataclass
class MarketClosenessResult:
    """Placeholder result container for a future extracted closeness pipeline."""

    config: MarketClosenessConfig
    pairs: pd.DataFrame
    metadata: dict[str, Any] | None = None


class MarketClosenessDetector:
    """Notebook-template placeholder for future closeness extraction."""

    def __init__(self, config: MarketClosenessConfig | None = None) -> None:
        self.config = config or MarketClosenessConfig()

    def fit(self, markets: pd.DataFrame, probabilities: pd.DataFrame) -> MarketClosenessResult:
        raise NotImplementedError(
            "Market closeness is still developed inside "
            "frozen_notebooks/2_ticker_closeness.ipynb."
        )


__all__ = [
    "MarketClosenessConfig",
    "MarketClosenessDetector",
    "MarketClosenessResult",
]
