"""Filesystem helpers for notebooks and scripts that need repository-aware imports."""

from __future__ import annotations

from pathlib import Path
import sys


def setup_root(start: str | Path | None = None, *, add_to_syspath: bool = True) -> Path:
    """Resolve the repository root from a starting path and optionally prepend it to ``sys.path``."""

    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "polymarket_research").exists():
            repo_root = candidate
            break
    else:
        raise RuntimeError(f"Could not locate repo root from cwd={current}")

    if add_to_syspath and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root
