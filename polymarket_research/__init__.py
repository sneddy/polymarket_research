"""Top-level package exports for the basic Polymarket dataset interface."""

from polymarket_research.data import (
    CanonicalDataset,
    CanonicalDatasetBuilder,
    DataBundle,
    DataPaths,
    ExternalShockConfig,
    MarketSelectionConfig,
    PanelBuildConfig,
    RawDatasetBundle,
    RawDatasetHandle,
    RawExternalCovariates,
    RawPolymarketBundle,
    RawPolymarketHandle,
    RawPolymarketDataset,
    RawPolymarketSnapshot,
    RawSnapshot,
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
    "RawDatasetBundle",
    "RawDatasetHandle",
    "RawExternalCovariates",
    "RawPolymarketBundle",
    "RawPolymarketHandle",
    "RawPolymarketDataset",
    "RawPolymarketSnapshot",
    "RawSnapshot",
]
