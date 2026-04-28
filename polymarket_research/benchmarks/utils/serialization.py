"""Serialization and bundle README helpers for frozen benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA_VERSION = 1


def to_json_ready(value: Any) -> Any:
    """Convert benchmark config values into JSON-safe objects."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_ready(v) for v in value]
    return value


def render_bundle_readme(
    *,
    title: str,
    summary_lines: list[str],
    manifest: dict[str, Any],
) -> str:
    """Render a compact README for a local frozen benchmark bundle."""
    return "\n".join(
        [
            f"# {title}",
            "",
            *summary_lines,
            "",
            "## Files",
            "- `manifest.json`",
            "- `examples.parquet`: leakage-safe observable input rows",
            "- `market_timeseries.parquet`",
            "- `targets.parquet`: labels and auxiliary target fields",
            "",
            "## Manifest",
            "```json",
            json.dumps(manifest, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
