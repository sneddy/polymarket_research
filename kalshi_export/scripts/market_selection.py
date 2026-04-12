from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kalshi_registry.schema import ensure_schema
from kalshi_registry.selection import rebuild_selected_markets
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build `selected_markets` from Kalshi `raw_markets`.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument(
        "--min-volume",
        type=float,
        default=20_000.0,
        help="Minimum `volume_num` required for a market to enter `selected_markets`. Default: 20000.",
    )
    p.add_argument(
        "--force-remove",
        action="store_true",
        help="Clear `selected_markets` before rebuilding it. By default the script preserves existing rows and upserts matching ids.",
    )
    p.add_argument(
        "--selection-version",
        default="v1_min_volume_only",
        help="Selection logic version string stored in `selected_markets`.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="market_selection",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        stats = rebuild_selected_markets(
            conn,
            min_volume=args.min_volume,
            force_remove=args.force_remove,
            selection_version=args.selection_version,
        )
        logger.info(
            "Kalshi market_selection finished | selected_rows=%s db=%s",
            stats["selected_rows"],
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
