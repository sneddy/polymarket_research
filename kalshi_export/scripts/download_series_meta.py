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

from clients.kalshi_client import KalshiClient
from kalshi_registry.refresh import refresh_raw_series
from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download the Kalshi series universe into `raw_series`.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument("--page-limit", type=int, default=200, help="Kalshi `/series` page size.")
    p.add_argument("--max-pages", type=int, default=None, help="Optional cap on `/series` pages to scan.")
    p.add_argument(
        "--force-remove",
        action="store_true",
        help="Clear `raw_series` before downloading. By default the script preserves existing rows and upserts new ones.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="download_series_meta",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)
    kalshi = KalshiClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        stats = refresh_raw_series(
            conn,
            kalshi=kalshi,
            limit=args.page_limit,
            max_pages=args.max_pages,
            force_remove=args.force_remove,
        )
        logger.info(
            "Kalshi download_series_meta finished | fetched_rows=%s total_raw_series_rows=%s db=%s",
            stats["fetched_rows"],
            stats["table_rows"],
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
