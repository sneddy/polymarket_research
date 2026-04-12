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

from clients.gamma_client import GammaClient
from polymarket_registry.refresh import refresh_market_universe
from polymarket_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH
from scripts.common import DEFAULT_LOG_DIR
from scripts.common import init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and upsert the broad Polymarket market universe metadata."
    )
    p.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="SQLite database path for the shared market registry.",
    )
    p.add_argument(
        "--min-created-at",
        default="2025-01-01T00:00:00Z",
        help="Lower bound for market createdAt when rebuilding the market registry.",
    )
    p.add_argument(
        "--max-metadata-pages",
        type=int,
        default=10,
        help="Maximum metadata pages to scan during universe refresh.",
    )
    p.add_argument(
        "--page-limit",
        type=int,
        default=1000,
        help="Gamma page size for market metadata requests.",
    )
    p.add_argument(
        "--include-active",
        action="store_true",
        help="Also include open/active markets; by default only closed markets are refreshed.",
    )
    p.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Preserve existing market_universe rows and upsert new rows instead of rebuilding the table from scratch.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    p.add_argument(
        "--log-dir",
        default=str(DEFAULT_LOG_DIR),
        help="Directory for per-run log files.",
    )
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="download_market_meta",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)

    gamma = GammaClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        universe_df = refresh_market_universe(
            conn,
            gamma=gamma,
            min_created_at=args.min_created_at,
            max_metadata_pages=args.max_metadata_pages,
            page_limit=args.page_limit,
            include_active=args.include_active,
            preserve_existing=args.preserve_existing,
        )
        logger.info(
            "market metadata download finished | total_universe_rows=%s db=%s",
            len(universe_df),
            db_path,
        )
        universe_count = conn.execute("SELECT COUNT(*) FROM market_universe").fetchone()[0]
        logger.info("market universe table count | rows=%s", universe_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
