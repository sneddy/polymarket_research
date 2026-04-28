"""HuggingFace Hub helpers for the terminal benchmark parquet bundle."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from polymarket_research.benchmarks.io.loaders import load_terminal
from polymarket_research.benchmarks.schemas.terminal import TerminalBenchmark
from polymarket_research.benchmarks.utils.sources import source_display_name


def _require_huggingface_hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra installed.
        raise ImportError(
            "huggingface_hub is required for HuggingFace export/import. "
            "Install the optional dependency with `pip install polymarket-research[hf]`."
        ) from exc
    return HfApi, snapshot_download


def _dataset_card(benchmark: TerminalBenchmark) -> str:
    manifest = benchmark.manifest()
    config = manifest["config"]
    horizons = ", ".join(str(value) for value in manifest["horizons_hours"]) or "none"
    split_counts = manifest["split_counts"]
    split_summary = ", ".join(f"{name}={count}" for name, count in split_counts.items()) or "none"
    release_name = manifest["release_name"]
    source = manifest.get("source", "polymarket")
    source_name = source_display_name(source)
    pretty_name = f"{source_name} Terminal {horizons}h Benchmark" if len(manifest["horizons_hours"]) == 1 else f"{source_name} Terminal Outcome Benchmark"

    summary = [
        f"# {pretty_name}",
        "",
        f"Frozen terminal-outcome benchmark built from {source_name} market histories.",
        "",
        "## Summary",
        f"- Source: `{source}`",
        f"- Release: `{release_name}`",
        f"- Examples: {manifest['rows']}",
        f"- Market-timeseries rows: {manifest['market_timeseries_rows']}",
        f"- Splits: {split_summary}",
        f"- Horizons (hours): {horizons}",
        f"- Split policy: `{config['split_on']}`",
        "",
        "## Files",
        "- `examples.parquet`: leakage-safe observable input rows",
        "- `market_timeseries.parquet`: normalized market-level probability histories keyed by `market_id`",
        "- `targets.parquet`: frozen labels and auxiliary target fields keyed by benchmark ids",
        "- `manifest.json`: benchmark metadata and build config",
        "",
        "## Build Manifest",
        "```json",
        json.dumps(manifest, indent=2, sort_keys=True),
        "```",
        "",
    ]

    return "\n".join(
        [
            "---",
            "license: cc-by-4.0",
            "task_categories:",
            "  - time-series-forecasting",
            "  - tabular-classification",
            "tags:",
            "  - prediction-markets",
            f"  - {source}",
            "  - forecasting",
            "  - benchmark",
            f"pretty_name: {pretty_name}",
            "---",
            "",
            *summary,
        ]
    )


def push_terminal_benchmark(
    benchmark: TerminalBenchmark,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Push a TerminalBenchmark to HuggingFace Hub as a dataset repo."""
    HfApi, _ = _require_huggingface_hub()
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir)
        benchmark.save(bundle_dir)
        (bundle_dir / "README.md").write_text(_dataset_card(benchmark), encoding="utf-8")

        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            private=private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(bundle_dir),
            token=token,
        )
    return str(repo_url)


def load_terminal_benchmark(
    repo_id: str,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> TerminalBenchmark:
    """Load a TerminalBenchmark from a HuggingFace Hub dataset repo."""
    _, snapshot_download = _require_huggingface_hub()
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        revision=revision,
        allow_patterns=["examples.parquet", "market_timeseries.parquet", "targets.parquet", "manifest.json"],
    )
    return load_terminal(local_dir)
