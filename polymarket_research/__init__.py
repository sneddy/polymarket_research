"""Top-level package exports for the basic Polymarket dataset interface."""

from polymarket_research.data import (
    CanonicalDataset,
    CanonicalDatasetBuilder,
    DataBundle,
    DataPaths,
    ExternalShockConfig,
    MarketSelectionConfig,
    PanelBuildConfig,
    RawExternalCovariates,
    RawPolymarketDataset,
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
    "RawExternalCovariates",
    "RawPolymarketDataset",
]
