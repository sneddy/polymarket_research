"""Top-level package exports for the basic Polymarket dataset interface."""

from polymarket_research.data import (
    CanonicalDataset,
    CanonicalDatasetBuilder,
    DataBundle,
    DataPaths,
    ExternalShockConfig,
    MarketSelectionConfig,
    PanelBuildConfig,
    PolymarketDatasetBuilder,
    RawExternalCovariates,
    RawPolymarketDataset,
    ResolvedMarketRepository,
)

PolymarketDataset = RawPolymarketDataset

__all__ = [
    "CanonicalDataset",
    "CanonicalDatasetBuilder",
    "DataBundle",
    "DataPaths",
    "ExternalShockConfig",
    "MarketSelectionConfig",
    "PanelBuildConfig",
    "PolymarketDataset",
    "PolymarketDatasetBuilder",
    "RawExternalCovariates",
    "RawPolymarketDataset",
    "ResolvedMarketRepository",
]
