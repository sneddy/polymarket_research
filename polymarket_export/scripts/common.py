from __future__ import annotations

import logging
from pathlib import Path


DEFAULT_DB_PATH = Path("db") / "resolved_probability_dataset.sqlite"
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
