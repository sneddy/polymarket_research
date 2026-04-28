"""Helpers for the on-disk frozen benchmark release layout."""

from __future__ import annotations

from pathlib import Path


DEFAULT_BENCHMARK_RELEASE_VERSION = "v1"
VALID_BENCHMARK_TASKS = frozenset({"terminal", "decisiveness", "repricing"})


def benchmark_release_root(repo_root: str | Path, source: str) -> Path:
    """Return the source-specific root that stores frozen benchmark releases."""
    return Path(repo_root) / "benchmark_releases" / str(source)


def benchmark_artifact_root(path: str | Path) -> Path:
    """Return the root directory that directly contains source release folders."""
    return Path(path)


def benchmark_bundle_dir(
    artifact_root: str | Path,
    *,
    source: str,
    task: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> Path:
    """Return a task bundle directory from a downloaded benchmark artifact root."""
    normalized_task = str(task)
    if normalized_task not in VALID_BENCHMARK_TASKS:
        valid = ", ".join(sorted(VALID_BENCHMARK_TASKS))
        raise ValueError(f"task must be one of: {valid}.")
    normalized_version = str(version).strip()
    if not normalized_version:
        raise ValueError("version must be a non-empty string.")
    return benchmark_artifact_root(artifact_root) / str(source) / normalized_task / normalized_version


def benchmark_release_dir(
    repo_root: str | Path,
    *,
    source: str,
    task: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> Path:
    """Return the frozen release directory for one benchmark task and version."""
    return benchmark_bundle_dir(
        Path(repo_root) / "benchmark_releases",
        source=source,
        task=task,
        version=version,
    )


def benchmark_release_report_dir(
    repo_root: str | Path,
    *,
    source: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> Path:
    """Return the directory that stores source-level release reports for one version."""
    normalized_version = str(version).strip()
    if not normalized_version:
        raise ValueError("version must be a non-empty string.")
    return benchmark_release_root(repo_root, source) / "reports" / normalized_version


def benchmark_report_dir(
    artifact_root: str | Path,
    *,
    source: str,
    version: str = DEFAULT_BENCHMARK_RELEASE_VERSION,
) -> Path:
    """Return a source report directory from a downloaded benchmark artifact root."""
    normalized_version = str(version).strip()
    if not normalized_version:
        raise ValueError("version must be a non-empty string.")
    return benchmark_artifact_root(artifact_root) / str(source) / "reports" / normalized_version
