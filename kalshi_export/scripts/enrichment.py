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

from clients.kalshi_client import KalshiClient
from kalshi_registry.enrichment import enrich_selected_markets
from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich Kalshi `selected_markets` with event-level metadata.")
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Refetch event metadata even when `event_metadata` already contains the event.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="enrichment",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)
    kalshi = KalshiClient()
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        stats = enrich_selected_markets(conn, kalshi=kalshi, refresh_existing=args.refresh_existing)
        logger.info(
            "Kalshi enrichment finished | fetched_events=%s upserted_event_rows=%s updated_selected_rows=%s upserted_universe_rows=%s db=%s",
            stats["fetched_events"],
            stats["upserted_event_rows"],
            stats["updated_selected_rows"],
            stats["upserted_universe_rows"],
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
