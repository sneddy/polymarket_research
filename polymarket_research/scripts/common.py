"""Shared helpers for script-style artifact builders."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Iterable

import pandas as pd

from polymarket_research.utils.filesystem import setup_root


VALID_SOURCES = ("polymarket", "kalshi")
VALID_BENCHMARK_TASKS = ("terminal", "decisiveness", "repricing")


def resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    """Resolve the repository root without mutating ``sys.path``."""
    return setup_root(repo_root, add_to_syspath=False)


def internal_cache_root(repo_root: str | Path, source: str) -> Path:
    """Return the internal cache root used by notebook and script workflows."""
    return Path(repo_root) / "frozen_notebooks" / "running_artefacts" / str(source)


def default_canonical_cache_dir(repo_root: str | Path, source: str) -> Path:
    """Return the default internal canonical artifact directory."""
    return internal_cache_root(repo_root, source) / "canonical_dataset"


def default_raw_snapshot_dir(repo_root: str | Path, source: str) -> Path:
    """Return the default internal raw snapshot directory."""
    return internal_cache_root(repo_root, source) / "raw"


def default_external_covariates_path(repo_root: str | Path) -> Path:
    """Return the default raw external-covariates root."""
    return Path(repo_root) / "cached_data" / "external_covariates"


def parse_optional_timestamp(value: str | None) -> pd.Timestamp | None:
    """Parse an optional UTC timestamp argument."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return pd.Timestamp(text)


def parse_csv_strings(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of non-empty strings."""
    if value is None:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_csv_ints(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated list of integers."""
    return tuple(int(item) for item in parse_csv_strings(value))


def parse_csv_floats(value: str | None) -> tuple[float, ...]:
    """Parse a comma-separated list of floats."""
    return tuple(float(item) for item in parse_csv_strings(value))


def parse_int_float_map(value: str | None) -> dict[int, float] | None:
    """Parse a comma-separated mapping like ``24=12,168=24``."""
    if value is None:
        return None

    mapping: dict[int, float] = {}
    for item in parse_csv_strings(value):
        if "=" not in item:
            raise ValueError("Expected key=value pairs like '24=12,168=24'.")
        key_text, value_text = item.split("=", 1)
        mapping[int(key_text.strip())] = float(value_text.strip())
    return mapping


def normalize_tasks(tasks: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a task iterable into a validated ordered tuple."""
    if tasks is None:
        return VALID_BENCHMARK_TASKS

    seen: list[str] = []
    for raw_task in tasks:
        task = str(raw_task).strip()
        if not task:
            continue
        if task == "all":
            return VALID_BENCHMARK_TASKS
        if task not in VALID_BENCHMARK_TASKS:
            valid = ", ".join(VALID_BENCHMARK_TASKS)
            raise ValueError(f"Unknown benchmark task {task!r}. Expected one of: {valid}.")
        if task not in seen:
            seen.append(task)
    return tuple(seen)


def print_frame(title: str, frame: pd.DataFrame) -> None:
    """Print a compact dataframe summary for script output."""
    print(title)
    if frame.empty:
        print("<empty>")
        return
    print(frame.to_string(index=False))


def log_message(prefix: str, message: str) -> None:
    """Print a normalized script log message."""
    print(f"{prefix} {message}")


@contextmanager
def log_stage(prefix: str, stage: str):
    """Log the start and completion time of a script stage."""
    log_message(prefix, f"start: {stage}")
    started_at = perf_counter()
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started_at
        log_message(prefix, f"failed: {stage} ({elapsed:.2f}s)")
        raise
    elapsed = perf_counter() - started_at
    log_message(prefix, f"done: {stage} ({elapsed:.2f}s)")
