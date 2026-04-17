from __future__ import annotations

import argparse
from datetime import UTC, datetime
import logging
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clients.kalshi_client import KalshiClient
from kalshi_registry.refresh import refresh_raw_markets_from_selected_series
from kalshi_registry.schema import ensure_schema
from scripts.common import DEFAULT_DB_PATH, DEFAULT_LOG_DIR, init_run_context


logger = logging.getLogger(__name__)


def _default_max_close_ts() -> int:
    now = datetime.now(tz=UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(today_start.timestamp())


def _parse_utc_date_start(value: str) -> int:
    dt = datetime.strptime(value.strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp())


def _resolve_close_date_range(args: argparse.Namespace) -> tuple[int | None, int]:
    min_close_ts = None if args.min_close_date is None else _parse_utc_date_start(args.min_close_date)
    max_close_ts = _default_max_close_ts() if args.max_close_date is None else _parse_utc_date_start(args.max_close_date)
    if min_close_ts is not None and min_close_ts > max_close_ts:
        raise ValueError("--min-close-date must be earlier than or equal to --max-close-date.")
    return min_close_ts, max_close_ts


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Kalshi raw markets for `selected_series` from live, historical, or both.")
    p.add_argument(
        "--source-mode",
        default="live",
        choices=["live", "historical", "both"],
        help="Which market branch to download for `selected_series`: live `/markets`, archived `/historical/markets`, or both.",
    )
    p.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="SQLite database path.")
    p.add_argument("--page-limit", type=int, default=200, help="Kalshi `/markets` page size.")
    p.add_argument(
        "--max-pages-per-series",
        type=int,
        default=None,
        help="Optional cap on `/markets` pages per selected series.",
    )
    p.add_argument(
        "--write-batch-pages",
        type=int,
        default=100,
        help="Number of fetched API pages to buffer before upserting into SQLite. Default: 100.",
    )
    p.add_argument("--status", default=None, help="Optional live `/markets` status filter.")
    p.add_argument("--min-close-date", default=None, help="Optional live lower bound on market close date in `YYYY-MM-DD`.")
    p.add_argument(
        "--max-close-date",
        default=None,
        help="Optional live upper bound on market close date in `YYYY-MM-DD`. Defaults to today in UTC.",
    )
    p.add_argument(
        "--include-mve",
        action="store_true",
        help="Include MVE/collection-like markets. Default: exclude them at the API layer.",
    )
    p.add_argument(
        "--force-remove",
        action="store_true",
        help="Clear `raw_markets` before downloading. By default the script preserves existing rows and upserts new ones.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory for per-run log files.")
    return p.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    min_close_ts, max_close_ts = _resolve_close_date_range(args)
    db_path, log_path = init_run_context(
        log_level=args.log_level,
        log_dir=args.log_dir,
        log_stem="download_market_meta",
        db_path=args.db_path,
    )
    logger.info("run logging initialized | log_path=%s", log_path)
    kalshi = KalshiClient()
    source_modes = ["markets"] if args.source_mode == "live" else ["historical"] if args.source_mode == "historical" else ["markets", "historical"]
    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        refresh_stats = None
        for idx, endpoint in enumerate(source_modes):
            refresh_stats = refresh_raw_markets_from_selected_series(
                conn,
                kalshi=kalshi,
                endpoint=endpoint,
                limit=args.page_limit,
                max_pages_per_series=args.max_pages_per_series,
                force_remove=args.force_remove and idx == 0,
                write_batch_pages=args.write_batch_pages,
                status=args.status,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                exclude_mve=not args.include_mve,
            )
            logger.info(
                "Kalshi raw markets source phase finished | endpoint=%s fetched_rows=%s total_raw_market_rows=%s selected_series_rows=%s attempted_series_rows=%s completed_series_rows=%s remaining_series_rows=%s",
                endpoint,
                refresh_stats["fetched_rows"],
                refresh_stats["table_rows"],
                refresh_stats["selected_series_rows"],
                refresh_stats.get("attempted_series_rows", 0),
                refresh_stats.get("completed_series_rows", 0),
                refresh_stats.get("remaining_series_rows", 0),
            )
        assert refresh_stats is not None
        logger.info(
            "Kalshi raw markets download finished | source_mode=%s fetched_rows=%s total_raw_market_rows=%s selected_series_rows=%s attempted_series_rows=%s completed_series_rows=%s remaining_series_rows=%s db=%s",
            args.source_mode,
            refresh_stats["fetched_rows"],
            refresh_stats["table_rows"],
            refresh_stats["selected_series_rows"],
            refresh_stats.get("attempted_series_rows", 0),
            refresh_stats.get("completed_series_rows", 0),
            refresh_stats.get("remaining_series_rows", 0),
            db_path,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
