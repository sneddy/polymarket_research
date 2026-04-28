"""Stable local-path loaders for frozen benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

from polymarket_research.benchmarks.io.paths import DEFAULT_BENCHMARK_RELEASE_VERSION, benchmark_bundle_dir
from polymarket_research.benchmarks.schemas.decisiveness import DecisivenessBenchmark
from polymarket_research.benchmarks.schemas.repricing import RepricingBenchmark
from polymarket_research.benchmarks.schemas.terminal import TerminalBenchmark


def load_terminal(path: str | Path) -> TerminalBenchmark:
    """Load a terminal benchmark bundle from a local directory."""
    return TerminalBenchmark.load(path)


def load_decisiveness(path: str | Path) -> DecisivenessBenchmark:
    """Load a decisiveness benchmark bundle from a local directory."""
    return DecisivenessBenchmark.load(path)


def load_repricing(path: str | Path) -> RepricingBenchmark:
    """Load a repricing benchmark bundle from a local directory."""
    return RepricingBenchmark.load(path)


def load_terminal_release(
    artifact_root: str | Path,
    *,
    source: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> TerminalBenchmark:
    """Load a terminal benchmark from a downloaded artifact root."""
    return load_terminal(benchmark_bundle_dir(artifact_root, source=source, task="terminal", version=version))


def load_decisiveness_release(
    artifact_root: str | Path,
    *,
    source: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> DecisivenessBenchmark:
    """Load a decisiveness benchmark from a downloaded artifact root."""
    return load_decisiveness(benchmark_bundle_dir(artifact_root, source=source, task="decisiveness", version=version))


def load_repricing_release(
    artifact_root: str | Path,
    *,
    source: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> RepricingBenchmark:
    """Load a repricing benchmark from a downloaded artifact root."""
    return load_repricing(benchmark_bundle_dir(artifact_root, source=source, task="repricing", version=version))
