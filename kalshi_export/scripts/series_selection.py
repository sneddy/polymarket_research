from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kalshi_registry.schema import ensure_schema
from kalshi_registry.series_selection import rebuild_selected_series
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build `selected_series` from `raw_series`.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument(
        "--force-remove",
        action="store_true",
        help="Clear `selected_series` before rebuilding it. By default the script preserves existing rows and upserts matching ids.",
    )
    p.add_argument(
        "--selection-version",
        default="v2_drop_short_frequencies_then_allow_categories_and_deny_short_term",
        help="Selection logic version string stored in `selected_series`.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="series_selection",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        stats = rebuild_selected_series(
            conn,
            force_remove=args.force_remove,
            selection_version=args.selection_version,
        )
        logger.info(
            "Kalshi series_selection finished | selected_rows=%s plot_path=%s db=%s",
            stats["selected_rows"],
            stats.get("plot_path"),
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
