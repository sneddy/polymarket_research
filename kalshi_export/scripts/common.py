from __future__ import annotations

from datetime import UTC, datetime
import logging
from pathlib import Path


DEFAULT_DB_PATH = Path("db") / "kalshi_probability_dataset.sqlite"
DEFAULT_LOG_DIR = Path("logs")


def setup_logging(level: str, *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper()))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def init_run_context(
    *,
    log_level: str,
    log_dir: str | Path,
    log_stem: str,
    db_path: str | Path,
) -> tuple[Path, Path]:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_log_dir = Path(log_dir).expanduser().resolve()
    log_path = resolved_log_dir / f"{log_stem}_{timestamp}.log"
    setup_logging(log_level, log_path=log_path)

    resolved_db_path = Path(db_path).expanduser().resolve()
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    return resolved_db_path, log_path
