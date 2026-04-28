"""Benchmark source-name helpers."""

from __future__ import annotations

from pathlib import Path


def normalize_source_name(source: str | None) -> str:
    """Normalize a benchmark source id for manifest and release-name use."""
    normalized = str(source or "polymarket").strip().lower().replace(" ", "_")
    return normalized or "polymarket"


def source_display_name(source: str | None) -> str:
    """Return a human-readable source name for bundle READMEs."""
    normalized = normalize_source_name(source)
    known_names = {
        "polymarket": "Polymarket",
        "kalshi": "Kalshi",
    }
    return known_names.get(normalized, normalized.replace("_", " ").title())


def infer_source_from_release_path(directory: str | Path, *, default: str = "polymarket") -> str:
    """Infer source from benchmark_releases/{source}/{task}/{version} when absent in old manifests."""
    parts = Path(directory).parts
    try:
        index = parts.index("benchmark_releases")
    except ValueError:
        return normalize_source_name(default)
    if index + 1 >= len(parts):
        return normalize_source_name(default)
    return normalize_source_name(parts[index + 1])
